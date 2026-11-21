//! 导出给前端的 `dist/brawl_meta.json` 精简结构。
//!
//! 体积优先：所有浮点指标保留 4 位小数，克制矩阵只保留达到阈值的对位。
//!
//! ## 两级聚合
//!
//! * `maps[]`  —— **地图维度**：出场率、平滑胜率、综合评分、选人顺位。
//!   评分是「该图内相对强度」，必须按图算。
//! * `modes[]` —— **模式维度**：1v1 克制矩阵。
//!   同一模式（如 `brawlBall`）下各图的克制关系高度一致，按模式聚合能把样本
//!   放大数倍；若按 24 张单图切碎，绝大多数对位样本会低于阈值而消失。
//!
//! 前端流程：选地图 -> 读 `maps[i]` 拿评分与顺位 -> 用该地图的 `mode` 去
//! `modes[]` 里取克制矩阵。
//!
//! ## 统计口径
//!
//! `games` / `wins` 是**原始计数**（便于展示样本量），
//! `win_rate` / `smoothed_wr` / `use_rate` 是**时间衰减加权后**的比率，
//! 因此 `wins / games` 不会精确等于 `win_rate`。

use serde::{Deserialize, Serialize};

/// 导出格式版本号，前端可据此做兼容处理。
///
/// v2：克制矩阵从地图维度迁移到模式维度，新增 `modes[]`，`Matchup` 增加 `smoothed_wr`。
pub const SCHEMA_VERSION: u32 = 2;

/// 导出根节点。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaOutput {
    pub schema_version: u32,
    /// 生成时间（RFC3339，UTC）。
    pub updated_at: String,
    /// 赛季起始日期 `YYYY-MM-DD`，早于此日期的对局已剔除。
    pub season_start: String,
    /// 参与统计的对局总数（原始计数）。
    pub total_battles: u64,
    /// 统计时使用的段位阈值。
    pub min_tier: i64,
    /// 统计参数，便于前端解释 score 与克制矩阵的口径。
    pub params: AnalysisParams,
    /// 英雄字典（id -> name），供前端把整数 ID 还原成可读名字。
    pub brawlers: Vec<BrawlerRef>,
    /// 按地图聚合的结果，按 `total_games` 降序。
    pub maps: Vec<MapMeta>,
    /// 按模式聚合的克制矩阵，按 `total_games` 降序。
    pub modes: Vec<ModeMeta>,
    /// 英雄定位标签配置（原样打包）。
    pub roles: RolesConfig,
}

/// 统计参数回显。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisParams {
    /// 胜率贝叶斯平滑的先验权重 m。
    pub bayes_prior_weight: f64,
    /// 克制矩阵贝叶斯平滑的先验权重 m（默认 6，先验胜率 0.5）。
    pub matchup_prior_weight: f64,
    /// 克制矩阵最小交锋样本量（原始场次）。
    pub matchup_min_samples: u32,
    /// 时间衰减半衰期（天），权重 `w = 0.5^(Δt / half_life)`。
    pub decay_half_life_days: f64,
}

/// 英雄字典项。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BrawlerRef {
    pub id: i64,
    pub name: String,
}

impl MetaOutput {
    /// 序列化为 JSON 字符串。默认紧凑输出（前端只关心体积）。
    pub fn to_json(&self, pretty: bool) -> anyhow::Result<String> {
        let json = if pretty {
            serde_json::to_string_pretty(self)?
        } else {
            serde_json::to_string(self)?
        };
        Ok(json)
    }
}

/// 单张地图的聚合结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapMeta {
    pub map_name: String,
    /// 地图所属模式（取自 `battle.mode`，同图多模式时取样本最多的一个）。
    /// 前端用它去 `modes[]` 里查克制矩阵。
    pub mode: String,
    pub total_games: u64,
    /// 按 `score` 降序排列的英雄榜。
    pub brawlers: Vec<MapBrawlerStat>,
}

/// 单个英雄在某张地图上的表现（不含克制矩阵）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapBrawlerStat {
    pub id: i64,
    pub name: String,
    /// 原始出场场次。
    pub games: u64,
    /// 原始胜场。
    pub wins: u64,
    /// 时间衰减加权胜率。
    pub win_rate: f64,
    /// 贝叶斯平滑后的加权胜率。
    pub smoothed_wr: f64,
    /// 时间衰减加权出场率（该图内归一，全部英雄之和 ≈ 6）。
    pub use_rate: f64,
    /// 综合评分，0.0 ~ 100.0（每张地图最高分英雄为 100）。
    pub score: f64,
    /// 选人顺位建议（依据同模式的克制矩阵 + 该图强度，按分位分配）。
    pub pick_order: PickOrder,
}

/// 单个模式的聚合结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeMeta {
    /// 模式标识，如 `brawlBall` / `bounty` / `knockout`。
    pub mode: String,
    pub total_games: u64,
    /// 按加权胜率降序排列。
    pub brawlers: Vec<ModeBrawlerStat>,
}

/// 单个英雄在某个模式下的表现（含克制矩阵）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeBrawlerStat {
    pub id: i64,
    pub name: String,
    /// 原始出场场次。
    pub games: u64,
    /// 原始胜场。
    pub wins: u64,
    /// 时间衰减加权胜率。
    pub win_rate: f64,
    /// 贝叶斯平滑后的加权胜率。
    pub smoothed_wr: f64,
    /// 时间衰减加权出场率。
    pub use_rate: f64,
    /// 克制对位，按原始交锋场次降序；仅包含样本量达标者。
    pub matchups: Vec<Matchup>,
}

/// 1v1 对位记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Matchup {
    /// 对手英雄 ID。
    pub opponent_id: i64,
    /// 双方敌对交锋的原始场次（用于展示样本量）。
    pub games: u64,
    /// 时间衰减加权胜率。
    pub win_rate: f64,
    /// 贝叶斯平滑后的加权胜率，前端做克制矩阵时应优先使用它。
    pub smoothed_wr: f64,
}

/// 选人顺位建议。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PickOrder {
    /// 万金油先手：基础强度高、最差对位也不吃亏。
    First,
    /// 中间位：常规对局可用。
    Middle,
    /// 后手康特：克制波动大，适合看阵容再拿。
    Last,
}

impl PickOrder {
    /// 便于日志与测试断言的字符串形式。
    pub fn as_str(self) -> &'static str {
        match self {
            PickOrder::First => "FIRST",
            PickOrder::Middle => "MIDDLE",
            PickOrder::Last => "LAST",
        }
    }
}

// ---------------------------------------------------------------------------
// 英雄定位配置（config/brawler_roles.json）
// ---------------------------------------------------------------------------

/// 英雄定位配置文件结构。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolesConfig {
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub roles: Vec<RoleGroup>,
}

/// 一个定位分组。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleGroup {
    pub key: String,
    pub label: String,
    #[serde(default)]
    pub description: String,
    /// 英雄英文名，导出时会被解析成 ID 一并输出。
    #[serde(default)]
    pub brawlers: Vec<String>,
    /// 导出时由 analyze 填充：解析到的英雄 ID（配置里的名字在字典中找不到则为空）。
    #[serde(default, skip_deserializing)]
    pub brawler_ids: Vec<i64>,
}

impl RolesConfig {
    /// 从 JSON 文件读取定位配置。
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("读取定位配置 {} 失败：{e}", path.display()))?;
        let cfg: Self = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("解析定位配置 {} 失败：{e}", path.display()))?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_order_serializes_as_screaming_snake_case() {
        assert_eq!(serde_json::to_string(&PickOrder::First).unwrap(), "\"FIRST\"");
        assert_eq!(serde_json::to_string(&PickOrder::Last).unwrap(), "\"LAST\"");
        assert_eq!(PickOrder::Middle.as_str(), "MIDDLE");
    }

    #[test]
    fn roles_config_parses_repo_template() {
        let cfg = RolesConfig::load(std::path::Path::new("config/brawler_roles.json"))
            .expect("仓库内的定位配置必须可解析");
        assert!(!cfg.roles.is_empty());
        let keys: Vec<&str> = cfg.roles.iter().map(|r| r.key.as_str()).collect();
        for required in ["TANK", "DAMAGE", "WALLBREAK", "THROWER", "ASSASSIN", "CONTROL"] {
            assert!(keys.contains(&required), "缺少定位分组 {required}");
        }
    }

    #[test]
    fn brawler_ids_are_not_expected_from_disk() {
        // brawler_ids 由 analyze 阶段填充，配置文件里不应出现，避免出现手写 ID 与真实字典不一致
        let raw = std::fs::read_to_string("config/brawler_roles.json").unwrap();
        assert!(!raw.contains("brawler_ids"));
    }
}
