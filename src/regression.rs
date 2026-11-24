//! 离线 3v3 回归克制系数（`dist/regression_matchups.json`）的加载与严格校验。
//!
//! 文件由 `ml/train_matchups.py` 的 `export_json()` 训练导出，结构与
//! `dist/brawl_meta.json` **不同**：
//!
//! ```text
//! meta                     训练元信息（trained_at / battles / bo3_clusters ...）
//! global_matchups          [[min_id, max_id, delta_pp], ...]
//! modes[].mode             模式名（bounty / brawlBall / gemGrab / heist / hotZone / knockout）
//! modes[].matchup_table    [[min_id, max_id, delta_pp], ...]
//! maps[].map_name          地图名（与 brawl.db 的 `map_name` 逐字一致）
//! maps[].matchup_table     [[min_id, max_id, delta_pp], ...]
//! ```
//!
//! 三个层级都是**稠密的全英雄对表**（`108 * 107 / 2 = 5778` 条），每条形如
//! `[16000000, 16000001, -1.99]`：`min_id < max_id`，`delta_pp` 是「`min_id` 相对
//! `max_id`」的百分点残差（`-1.99` = `-1.99%`），保留 2 位小数。
//!
//! ## 为什么回归文件里没有 `n_eff`
//!
//! 样本量是**本地数据的事实**，不是模型参数：`n_eff`（时间衰减 × BO3 系列降权）与
//! 三层查表门槛（地图 `>= 5.0` / 模式 `>= 8.0` / 全局 `>= 6.0`）必须继续由 Rust
//! 侧计算。回归文件只替换「残差结论」，**不决定某对英雄有没有资格出现在输出里**——
//! 否则回归表的稠密性会把本地一两条样本的对位硬塞进查表，口径直接崩掉。
//!
//! ## 严格校验
//!
//! 读取失败、JSON 结构不符、`min_id >= max_id`、`delta_pp` 非有限数（NaN / inf）、
//! 同一层同一作用域出现重复对位键，一律 `Err` 直接退出，**绝不静默忽略**：
//! 一份半损坏的回归表会静默污染三层查表，事后无法从导出文件反查。

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

/// 规范方向对位键：`(min_id, max_id)`，恒有 `min_id < max_id`。
pub type PairKey = (i64, i64);

/// 回归系数所在的层级。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegLevel {
    /// 地图级：`maps[].matchup_table`。
    Map,
    /// 模式级：`modes[].matchup_table`。
    Mode,
    /// 全局级：`global_matchups`。
    Global,
}

impl RegLevel {
    /// 终端输出用的中文层级名。
    pub fn label(self) -> &'static str {
        match self {
            RegLevel::Map => "地图级",
            RegLevel::Mode => "模式级",
            RegLevel::Global => "全局级",
        }
    }
}

/// 训练元信息（`meta`）。
///
/// 只反序列化需要回显 / 做合理性判断的字段；`holdout_metrics`、`inner_selection`
/// 等结构随训练脚本演进频繁变动，强类型化只会让加载器跟着训练脚本一起碎掉。
#[derive(Debug, Clone, Deserialize)]
pub struct RegressionMeta {
    /// 训练完成时间（RFC3339）。
    pub trained_at: String,
    /// 参与训练的原始对局数。
    pub battles: u64,
    /// 有效 BO3 系列簇数（聚簇降权后的独立 BP 组数）。
    pub bo3_clusters: u64,
    /// 折算后的总有效权重（≈ `bo3_clusters`）。
    #[serde(default)]
    pub effective_weight: f64,
    /// 时间衰减半衰期（天），用于核对训练与推理口径是否一致。
    #[serde(default)]
    pub time_decay_half_life_days: f64,
    /// 贝叶斯平滑先验权重 m。
    #[serde(default)]
    pub bayes_prior_weight: f64,
    /// 主效应层级：`map_aware` / `global_only`。
    #[serde(default)]
    pub stage1_main_effect: String,
}

/// 单层的命中 / 回退计数。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LevelStats {
    /// 命中：回归文件里存在该键，`delta_pp` 被替换。
    pub hits: u64,
    /// 回退：回归文件缺该键（或整层缺失），沿用 Rust 原计算值。
    pub fallbacks: u64,
}

impl LevelStats {
    /// 该层参与统计的条目总数。
    pub fn total(&self) -> u64 {
        self.hits + self.fallbacks
    }

    /// 命中率（0.0 ~ 100.0），无条目时为 0。
    pub fn hit_rate(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            0.0
        } else {
            self.hits as f64 * 100.0 / total as f64
        }
    }
}

/// 三层的命中 / 回退汇总。
#[derive(Debug, Clone, Copy, Default)]
pub struct RegressionStats {
    pub map: LevelStats,
    pub mode: LevelStats,
    pub global: LevelStats,
}

impl RegressionStats {
    /// 三层合计命中数。
    pub fn total_hits(&self) -> u64 {
        self.map.hits + self.mode.hits + self.global.hits
    }

    /// 三层合计回退数。
    pub fn total_fallbacks(&self) -> u64 {
        self.map.fallbacks + self.mode.fallbacks + self.global.fallbacks
    }
}

/// 已校验并建索引的回归克制系数表。
///
/// 查表是**纯读**的（可跨 rayon 线程共享），命中 / 回退计数用原子变量累加——
/// 地图维度的导出是并行的，统计必须在 `analyze` 返回后仍是完整值。
#[derive(Debug)]
pub struct RegressionMatchups {
    /// 训练元信息。
    pub meta: RegressionMeta,
    /// 全局层：规范方向对位 -> `delta_pp`。
    global: HashMap<PairKey, f64>,
    /// 模式层：模式名 -> (规范方向对位 -> `delta_pp`)。
    modes: HashMap<String, HashMap<PairKey, f64>>,
    /// 地图层：地图名 -> (规范方向对位 -> `delta_pp`)。
    maps: HashMap<String, HashMap<PairKey, f64>>,
    /// 命中 / 回退计数器。
    counters: Counters,
}

/// 分层的原子计数器（地图导出是并行的，只能用原子变量）。
#[derive(Debug, Default)]
struct Counters {
    map_hits: AtomicU64,
    map_fallbacks: AtomicU64,
    mode_hits: AtomicU64,
    mode_fallbacks: AtomicU64,
    global_hits: AtomicU64,
    global_fallbacks: AtomicU64,
}

impl Counters {
    fn hit(&self, level: RegLevel) {
        let cell = match level {
            RegLevel::Map => &self.map_hits,
            RegLevel::Mode => &self.mode_hits,
            RegLevel::Global => &self.global_hits,
        };
        cell.fetch_add(1, Ordering::Relaxed);
    }

    fn fallback(&self, level: RegLevel) {
        let cell = match level {
            RegLevel::Map => &self.map_fallbacks,
            RegLevel::Mode => &self.mode_fallbacks,
            RegLevel::Global => &self.global_fallbacks,
        };
        cell.fetch_add(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// 反序列化用的原始结构
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawFile {
    meta: RegressionMeta,
    #[serde(default)]
    global_matchups: Vec<RawPair>,
    #[serde(default)]
    modes: Vec<RawMode>,
    #[serde(default)]
    maps: Vec<RawMap>,
}

#[derive(Debug, Deserialize)]
struct RawMode {
    mode: String,
    #[serde(default)]
    matchup_table: Vec<RawPair>,
}

#[derive(Debug, Deserialize)]
struct RawMap {
    map_name: String,
    #[serde(default)]
    matchup_table: Vec<RawPair>,
}

/// `[min_id, max_id, delta_pp]` 三元组。
type RawPair = (i64, i64, f64);

// ---------------------------------------------------------------------------
// 加载 + 校验
// ---------------------------------------------------------------------------

impl RegressionMatchups {
    /// 从磁盘读取回归系数文件。
    ///
    /// 读取或解析失败一律返回 `Err`（调用方必须直接退出），不做任何兜底。
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("读取回归系数文件 {} 失败", path.display()))?;
        Self::from_json_str(&raw)
            .with_context(|| format!("解析回归系数文件 {} 失败", path.display()))
    }

    /// 从 JSON 字符串构建（供测试与 `load` 复用）。
    pub fn from_json_str(raw: &str) -> Result<Self> {
        let file: RawFile = serde_json::from_str(raw).context(
            "JSON 结构不符合 regression_matchups 格式（期望 meta / global_matchups / modes / maps）",
        )?;

        let global = index_table(RegLevel::Global, "", file.global_matchups)?;

        let mut modes: HashMap<String, HashMap<PairKey, f64>> =
            HashMap::with_capacity(file.modes.len());
        for item in file.modes {
            if item.mode.trim().is_empty() {
                bail!("modes[] 中存在空模式名");
            }
            let table = index_table(RegLevel::Mode, &item.mode, item.matchup_table)?;
            if modes.insert(item.mode.clone(), table).is_some() {
                bail!("modes[] 模式名重复：`{}`", item.mode);
            }
        }

        let mut maps: HashMap<String, HashMap<PairKey, f64>> =
            HashMap::with_capacity(file.maps.len());
        for item in file.maps {
            if item.map_name.trim().is_empty() {
                bail!("maps[] 中存在空地图名");
            }
            let table = index_table(RegLevel::Map, &item.map_name, item.matchup_table)?;
            if maps.insert(item.map_name.clone(), table).is_some() {
                bail!("maps[] 地图名重复：`{}`", item.map_name);
            }
        }

        if global.is_empty() && modes.is_empty() && maps.is_empty() {
            bail!("回归系数文件里没有任何对位条目（global_matchups / modes / maps 全为空）");
        }

        Ok(Self {
            meta: file.meta,
            global,
            modes,
            maps,
            counters: Counters::default(),
        })
    }

    /// 取某层某作用域下规范方向 `(min_id, max_id)` 的回归 `delta_pp`。
    ///
    /// * `scope`：地图层传地图名、模式层传模式名，全局层忽略；
    /// * 该层 / 该作用域不存在，或对位键缺失，均返回 `None`（调用方回退 Rust 原值）。
    pub fn delta_pp(&self, level: RegLevel, scope: &str, min_id: i64, max_id: i64) -> Option<f64> {
        let table = match level {
            RegLevel::Global => &self.global,
            RegLevel::Mode => self.modes.get(scope)?,
            RegLevel::Map => self.maps.get(scope)?,
        };
        table.get(&(min_id, max_id)).copied()
    }

    /// 记一次命中。
    pub fn record_hit(&self, level: RegLevel) {
        self.counters.hit(level);
    }

    /// 记一次回退。
    pub fn record_fallback(&self, level: RegLevel) {
        self.counters.fallback(level);
    }

    /// 三层的命中 / 回退快照。必须在 `analyze` 返回后调用（并行段已 join）。
    pub fn stats(&self) -> RegressionStats {
        RegressionStats {
            map: LevelStats {
                hits: self.counters.map_hits.load(Ordering::Relaxed),
                fallbacks: self.counters.map_fallbacks.load(Ordering::Relaxed),
            },
            mode: LevelStats {
                hits: self.counters.mode_hits.load(Ordering::Relaxed),
                fallbacks: self.counters.mode_fallbacks.load(Ordering::Relaxed),
            },
            global: LevelStats {
                hits: self.counters.global_hits.load(Ordering::Relaxed),
                fallbacks: self.counters.global_fallbacks.load(Ordering::Relaxed),
            },
        }
    }

    /// 全局层条目数。
    pub fn global_len(&self) -> usize {
        self.global.len()
    }

    /// 模式层覆盖的模式数。
    pub fn mode_count(&self) -> usize {
        self.modes.len()
    }

    /// 地图层覆盖的地图数。
    pub fn map_count(&self) -> usize {
        self.maps.len()
    }
}

/// 把三元组数组建索引，并逐条做严格校验。
fn index_table(level: RegLevel, scope: &str, rows: Vec<RawPair>) -> Result<HashMap<PairKey, f64>> {
    let where_ = if scope.is_empty() {
        level.label().to_string()
    } else {
        format!("{} `{scope}`", level.label())
    };

    let mut out: HashMap<PairKey, f64> = HashMap::with_capacity(rows.len());
    for (idx, (min_id, max_id, delta_pp)) in rows.into_iter().enumerate() {
        if min_id >= max_id {
            bail!("{where_} 第 {idx} 条对位键非法：要求 min_id < max_id，实际 ({min_id}, {max_id})");
        }
        if !delta_pp.is_finite() {
            bail!("{where_} 对位 ({min_id}, {max_id}) 的 delta_pp 非有限数：{delta_pp}");
        }
        let key = (min_id, max_id);
        if out.contains_key(&key) {
            bail!("{where_} 对位 ({min_id}, {max_id}) 重复出现，无法确定采用哪一条");
        }
        // 统一保留 2 位小数（与导出精度一致），并归一 -0.0
        out.insert(key, round2(delta_pp));
    }
    Ok(out)
}

/// 保留 2 位小数，并把 `-0.0` 归一成 `0.0`。
fn round2(v: f64) -> f64 {
    let r = (v * 100.0).round() / 100.0;
    if r == 0.0 {
        0.0
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 最小可用样本：三层各一条，模式 / 地图作用域各不相同。
    fn sample_json() -> &'static str {
        r#"{
            "meta": {
                "trained_at": "2026-11-24T12:18:16+00:00",
                "battles": 1313527,
                "bo3_clusters": 556025,
                "effective_weight": 530627.36,
                "time_decay_half_life_days": 30.0,
                "bayes_prior_weight": 50.0,
                "stage1_main_effect": "map_aware"
            },
            "global_matchups": [[1, 2, 3.456], [1, 3, -0.001]],
            "modes": [
                {"mode": "brawlBall", "matchup_table": [[1, 2, -4.2]]}
            ],
            "maps": [
                {"map_name": "Center Stage", "mode": "brawlBall", "matchup_table": [[1, 2, 5.555]]}
            ]
        }"#
    }

    #[test]
    fn parses_three_levels_and_rounds_to_two_decimals() {
        let reg = RegressionMatchups::from_json_str(sample_json()).expect("样本必须可解析");

        assert_eq!(reg.meta.battles, 1_313_527);
        assert_eq!(reg.meta.bo3_clusters, 556_025);
        assert_eq!(reg.meta.trained_at, "2026-11-24T12:18:16+00:00");

        assert_eq!(reg.global_len(), 2);
        assert_eq!(reg.mode_count(), 1);
        assert_eq!(reg.map_count(), 1);

        // 三层分别取到各自的表，互不串味
        assert_eq!(reg.delta_pp(RegLevel::Global, "", 1, 2), Some(3.46));
        assert_eq!(reg.delta_pp(RegLevel::Mode, "brawlBall", 1, 2), Some(-4.2));
        assert_eq!(reg.delta_pp(RegLevel::Map, "Center Stage", 1, 2), Some(5.56));

        // 缺失的作用域 / 对位键必须返回 None（调用方据此回退）
        assert_eq!(reg.delta_pp(RegLevel::Map, "不存在的图", 1, 2), None);
        assert_eq!(reg.delta_pp(RegLevel::Mode, "heist", 1, 2), None);
        assert_eq!(reg.delta_pp(RegLevel::Global, "", 2, 1), None, "只认规范方向");
    }

    #[test]
    fn rejects_invalid_pair_key() {
        let raw = r#"{"meta":{"trained_at":"x","battles":1,"bo3_clusters":1},
                      "global_matchups":[[5,5,1.0]]}"#;
        let err = RegressionMatchups::from_json_str(raw).unwrap_err().to_string();
        assert!(err.contains("min_id < max_id"), "错误信息应指出键序问题，实际 {err}");

        let raw = r#"{"meta":{"trained_at":"x","battles":1,"bo3_clusters":1},
                      "global_matchups":[[9,3,1.0]]}"#;
        assert!(RegressionMatchups::from_json_str(raw).is_err());
    }

    #[test]
    fn rejects_non_finite_delta() {
        // 1e999 溢出成 inf：必须被拦下，绝不能带着 inf 去算分位数
        let raw = r#"{"meta":{"trained_at":"x","battles":1,"bo3_clusters":1},
                      "global_matchups":[[1,2,1e999]]}"#;
        assert!(RegressionMatchups::from_json_str(raw).is_err());
    }

    #[test]
    fn rejects_broken_json_and_duplicate_keys() {
        assert!(RegressionMatchups::from_json_str("not json").is_err());
        assert!(RegressionMatchups::from_json_str(r#"{"meta":{}}"#).is_err());

        let dup = r#"{"meta":{"trained_at":"x","battles":1,"bo3_clusters":1},
                     "global_matchups":[[1,2,1.0],[1,2,2.0]]}"#;
        let err = RegressionMatchups::from_json_str(dup).unwrap_err().to_string();
        assert!(err.contains("重复"), "重复键必须报错，实际 {err}");
    }

    #[test]
    fn rejects_empty_table() {
        let raw = r#"{"meta":{"trained_at":"x","battles":1,"bo3_clusters":1}}"#;
        assert!(RegressionMatchups::from_json_str(raw).is_err());
    }

    #[test]
    fn counters_are_per_level() {
        let reg = RegressionMatchups::from_json_str(sample_json()).unwrap();
        reg.record_hit(RegLevel::Map);
        reg.record_hit(RegLevel::Map);
        reg.record_fallback(RegLevel::Map);
        reg.record_hit(RegLevel::Global);

        let stats = reg.stats();
        assert_eq!(stats.map, LevelStats { hits: 2, fallbacks: 1 });
        assert_eq!(stats.mode, LevelStats::default());
        assert_eq!(stats.global, LevelStats { hits: 1, fallbacks: 0 });
        assert_eq!(stats.total_hits(), 3);
        assert_eq!(stats.total_fallbacks(), 1);
        assert!((stats.map.hit_rate() - 200.0 / 3.0).abs() < 1e-9);
        assert_eq!(stats.mode.hit_rate(), 0.0);
    }

    #[test]
    fn repo_regression_file_loads_when_present() {
        // 仓库内的产物若存在，必须能按新格式完整加载（不存在的环境跳过）
        let path = Path::new("dist/regression_matchups.json");
        if !path.exists() {
            return;
        }
        let reg = RegressionMatchups::load(path).expect("仓库内的回归系数文件必须可解析");
        assert!(reg.global_len() > 0);
        assert!(reg.map_count() > 0);
        assert!(reg.meta.bo3_clusters > 0);
    }
}
