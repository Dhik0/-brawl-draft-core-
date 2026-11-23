//! 导出给前端的 `dist/brawl_meta.json` 精简结构。
//!
//! 体积优先：所有浮点指标保留 4 位小数，克制关系保留 2 位小数且按强度截断。
//!
//! ## 三级聚合
//!
//! * `maps[]`  —— **地图维度**：出场率、平滑胜率、综合评分、选人顺位，以及
//!   **地图级克制列表**（`brawlers[].matchups`）。地形差异（草图 vs 开阔图）
//!   直接体现在这一层，是前端弹窗的首选数据源。
//! * `modes[]` —— **模式维度**：同一模式（如 `brawlBall`）下各图的克制关系高度一致，
//!   按模式聚合能把样本放大数倍，作为地图样本不足时的**先验**。
//! * `global_matchups[]` —— **全局维度**（v4 新增）：全库 1v1 去偏残差，跨模式兜底。
//!   它**不按 `|delta|` 剪枝**，均势对位也会导出，前端因此能区分「五五开」与
//!   「无数据」，3×3 对位表不再出现大面积空白。
//!
//! ## 克制口径
//!
//! `delta` 是 **地图分层去偏后的 Log5 超额胜场残差**（单位：百分点，保留 2 位小数），
//! 不是原始胜率：
//!
//! ```text
//! 本图      E_ab,m = log5(p_a,m, p_b,m)        // 用**该图**基础胜率算预期
//!           e1     = w_ab,m - n_ab,m * E_ab,m
//! 同模式其他图 e2    = E_mode - e1                // 已扣除本图，不重复计数
//! 其他模式    e3    = E_global - E_mode           // 已扣除本模式，不重复计数
//! 折算      n2_eff = min(0.6*n2, 40), n3_eff = min(0.3*n3, 25)
//! delta    = (e1 + e2_eff + e3_eff) / (n1 + n2_eff + n3_eff + 20) * 100
//! ```
//!
//! 这样扣掉了「英雄本身强度」带来的胜率红利，只留下**对位关系本身**的贡献，
//! 且按图分层避免了跨图混合的辛普森悖论。`delta > 0` 表示 A 在对位上占优，
//! `delta < 0` 表示 A 被克制，且严格反对称（`delta_BA = -delta_AB`）。
//!
//! 各层门槛：地图级 `eff_n >= 8`、模式级 `eff_n >= 12`，都要求 `|delta| >= 1.5pp`；
//! 全局兜底表只要 `n >= 10`，不剪枝。**后端不做「正残差过大就抹零」的处理**，
//! 绿灯绝对胜率由前端结合本图基础胜率自行兜底。
//!
//! ## 统计口径
//!
//! `games` / `wins` / `total_games` 是**原始计数**（便于展示样本量），
//! `win_rate` / `smoothed_wr` / `use_rate` 是**时间衰减 × BO3 系列降权后**的比率，
//! 因此 `wins / games` 不会精确等于 `win_rate`。
//!
//! 传奇排位是 BO3：同一批 6 人在 10 分钟内连打 2~3 小局，若每个小局都算独立样本，
//! 有效样本会虚高 2~3 倍。`Matchup.eff_samples` 因此是降权后的有效样本
//! （一轮 3 局 BO3 合计 `1.0`），`Matchup.samples` 仍是原始场次。

use serde::{Deserialize, Serialize};

/// 导出格式版本号，前端可据此做兼容处理。
///
/// * v2：克制矩阵从地图维度迁移到模式维度，`Matchup` 增加 `smoothed_wr`。
/// * v3：克制改为 **Log5 残差 delta**（`opponent` / `delta` / `samples`），
///   地图级也导出克制列表；新增段位审计 `tier_audit`。
/// * v4：新增全局 1v1 兜底表 `global_matchups`（不按 `|delta|` 剪枝）；收缩强度
///   `K` 由 50 降到 20，地图级 / 模式级门槛拆分为 `eff_n >= 8` / `12`，
///   显著性门槛降为 `|delta| >= 1.5pp`。v3 的字段名全部保留。
///   v4 内追加（未升版，纯增量）：`Matchup.eff_samples` 给出 BO3 系列降权后的
///   有效样本，`delta` 与 `eff_n` 门槛同步改用降权口径。
pub const SCHEMA_VERSION: u32 = 4;

/// 导出根节点。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaOutput {
    pub schema_version: u32,
    /// 生成时间（RFC3339，UTC）。
    pub updated_at: String,
    /// 赛季起始日期 `YYYY-MM-DD`，早于此日期的对局已剔除。
    pub season_start: String,
    /// 参与统计的对局总数（原始计数，已应用段位过滤）。
    pub total_battles: u64,
    /// 统计时使用的段位阈值。
    pub min_tier: i64,
    /// 段位审计：过滤前后对局数 + 各段位分布。
    pub tier_audit: TierAudit,
    /// 统计参数，便于前端解释 score 与克制矩阵的口径。
    pub params: AnalysisParams,
    /// 英雄字典（id -> name），供前端把整数 ID 还原成可读名字。
    pub brawlers: Vec<BrawlerRef>,
    /// 按地图聚合的结果，按 `total_games` 降序。
    pub maps: Vec<MapMeta>,
    /// 按模式聚合的克制矩阵，按 `total_games` 降序。
    pub modes: Vec<ModeMeta>,
    /// 全局 1v1 兜底表（v4）：全库去偏残差，跨模式兜底。
    ///
    /// 每对英雄只存一条（`a < b`），**不按 `|delta|` 剪枝**——`delta ≈ 0` 表示
    /// 均势对位，与「查不到 = 无数据」有明确区别。按 `(a, b)` 升序排列。
    pub global_matchups: Vec<GlobalMatchup>,
    /// 英雄定位标签配置（原样打包）。
    pub roles: RolesConfig,
}

/// 段位审计。
///
/// `avg_tier` 来自官方 API 的 `brawler.rank` / `player.rank`。该字段在实际
/// battlelog 响应中经常缺失，缺失时入库为 `0`，因此本结构把「未知段位」
/// 单独计数——`min_tier > 0` 但 `unknown_tier_battles == battles_total` 时，
/// 说明段位门槛实际没有生效。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierAudit {
    /// 配置的段位阈值。
    pub min_tier: i64,
    /// 过滤前总对局数。
    pub battles_total: u64,
    /// 过滤后保留的对局数。
    pub battles_kept: u64,
    /// 段位未知（`avg_tier <= 0`）的对局数。
    pub unknown_tier_battles: u64,
    /// 各段位的对局数分布（按段位升序，含 0 段位）。
    pub distribution: Vec<TierBucket>,
}

impl TierAudit {
    /// 段位信息是否可用于判定门槛。
    ///
    /// 注意语义：门槛在**入库**阶段就已生效（`FilterReason::BelowTier`），
    /// 所以 analyze 阶段「过滤前后局数相同」是正常现象，不代表门槛失效。
    /// 只有「全部对局的段位都不可解析」才说明门槛确实无从判定。
    pub fn filter_effective(&self) -> bool {
        self.min_tier <= 0
            || self.battles_total == 0
            || self.unknown_tier_battles < self.battles_total
    }
}

/// 单个段位的对局数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierBucket {
    pub tier: i64,
    /// 段位可读名，如「传奇 II」「职业」「未知」。
    pub label: String,
    pub battles: u64,
}

/// 排位段位分组名（索引 = `(tier - 1) / 3`）。
const TIER_GROUPS: [&str; 8] = [
    "青铜", "白银", "黄金", "钻石", "神话", "传奇", "大师", "职业",
];

/// 段位所属大段（青铜 / 白银 / 黄金 / 钻石 / 神话 / 传奇 / 大师 / 职业）。
pub fn tier_group(tier: i64) -> &'static str {
    match tier {
        1..=3 => TIER_GROUPS[0],
        4..=6 => TIER_GROUPS[1],
        7..=9 => TIER_GROUPS[2],
        10..=12 => TIER_GROUPS[3],
        13..=15 => TIER_GROUPS[4],
        16..=18 => TIER_GROUPS[5],
        19..=21 => TIER_GROUPS[6],
        22 => TIER_GROUPS[7],
        _ => "未知",
    }
}

/// 段位可读名：`16 -> "传奇 I"`、`22 -> "职业"`、`0 -> "未知"`。
pub fn tier_label(tier: i64) -> String {
    if tier <= 0 {
        return "未知".to_string();
    }
    if tier == 22 {
        return "职业".to_string();
    }
    if !(1..=21).contains(&tier) {
        return format!("未知({tier})");
    }
    let idx = ((tier - 1) / 3) as usize;
    let roman = ["I", "II", "III"][((tier - 1) % 3) as usize];
    format!("{} {}", TIER_GROUPS[idx], roman)
}

/// 统计参数回显。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisParams {
    /// 胜率贝叶斯平滑的先验权重 m。
    pub bayes_prior_weight: f64,
    /// 分层贝叶斯收缩的先验强度 K（伪场次）。
    ///
    /// 字段名为 v3 兼容保留（旧版是 `C = 15`），语义已统一为分层收缩强度。
    pub matchup_smoothing_c: f64,
    /// 导出克制条目的最小**有效样本** `eff_n`（旧字段名，语义已从原始场次改为 `eff_n`）。
    /// v4 起等价于地图级门槛（模式级见 `mode_matchup_min_eff_samples`）。
    pub matchup_min_samples: u32,
    /// 地图级克制的最小有效样本（与 `matchup_min_samples` 同值）。
    pub map_matchup_min_samples: u32,
    /// v3 兼容保留：旧版「单图场次达到该值后完全信任单图」的阈值，
    /// 现由互斥分层加权取代，仅作回显。
    pub map_matchup_trust_samples: u32,
    /// 时间衰减半衰期（天），权重 `w = 0.5^(Δt / half_life)`。
    pub decay_half_life_days: f64,
    /// 有效样本门槛（v3 兼容字段，等价于地图级门槛）。
    pub matchup_min_eff_samples: f64,
    /// 残差显著性门槛（`|delta| >=` 该值才导出，单位与计算口径一致：概率）。
    pub matchup_min_abs_delta: f64,
    /// 同模式其他地图的样本折算权重。
    pub same_mode_weight: f64,
    /// 同模式其他地图折算后的样本上限。
    pub same_mode_cap: f64,
    /// 其他模式的样本折算权重。
    pub other_mode_weight: f64,
    /// 其他模式折算后的样本上限。
    pub other_mode_cap: f64,
    /// v4：模式级克制的最小有效样本 `eff_n`。
    pub mode_matchup_min_eff_samples: f64,
    /// v4：全局兜底表的最小原始样本量 `n`。
    pub global_matchup_min_samples: u64,
    /// v4：全局兜底表的残差门槛，固定 `0.0` 表示**不按 `|delta|` 剪枝**。
    pub global_matchup_min_abs_delta: f64,
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
    /// 前端用它去 `modes[]` 里查模式级克制先验。
    pub mode: String,
    pub total_games: u64,
    /// 按 `score` 降序排列的英雄榜。
    pub brawlers: Vec<MapBrawlerStat>,
}

/// 单个英雄在某张地图上的表现（含该图的克制列表）。
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
    /// 该图内的克制列表：前 K 个优势对位（正 delta）+ 前 K 个劣势对位（负 delta），
    /// 按 `delta` 降序。已按 `eff_n` / 显著性过滤，前端弹窗可直接渲染。
    pub matchups: Vec<Matchup>,
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

/// 单个英雄在某个模式下的表现（含模式级克制矩阵）。
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
    /// 模式级克制：按 |delta| 取最显著的 K 条，再按 `delta` 降序输出。
    pub matchups: Vec<Matchup>,
}

/// 1v1 对位记录（地图分层去偏后的 Log5 残差口径）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Matchup {
    /// 对手英雄 ID。
    pub opponent: i64,
    /// 净胜率差值（百分点，2 位小数）。正数 = 该英雄占优，负数 = 被克制。
    /// 严格反对称：A 对 B 的 delta 恒等于 B 对 A 的 delta 的相反数。
    pub delta: f64,
    /// 真实对位样本场次（原始计数，未加权）。
    ///
    /// 注意：导出过滤用的是**折算后的有效样本** `eff_n`（本图 + 同模式其他图 +
    /// 其他模式的加权和），所以 `samples` 可能小于门槛而条目仍然存在——
    /// 那说明结论主要来自同模式 / 全库先验，本图只是佐证。
    pub samples: u64,
    /// 有效对位样本（时间衰减 + BO3 系列降权后）。
    ///
    /// 一轮 BO3 的 3 个小局会各乘 `1/3`，因此 3 局同阵容对局在这里合计为 `1.0`。
    /// `delta` 的分子分母都用这个口径，`eff_n` 门槛也按它判定。
    pub eff_samples: f64,
}

/// 全局 1v1 兜底记录（v4 新增，紧凑数组格式）。
///
/// JSON 里序列化成 **4 元数组** `[a, b, delta, n]`（不是对象）：
///
/// ```text
/// [1600001, 1600002, -1.25, 37]
///  id_a      id_b     delta  n
/// ```
///
/// * 无序存储：恒有 `a < b`，每对英雄只有一条记录；`delta` 的方向是「`a` 对 `b`」，
///   查 `b` 对 `a` 取相反数即可；
/// * 用数组而非对象是为了控体积：108 名英雄约 4400 条记录，对象格式要多花 ~80KB。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(into = "(i64, i64, f64, u64)", from = "(i64, i64, f64, u64)")]
pub struct GlobalMatchup {
    /// 较小的英雄 ID。
    pub a: i64,
    /// 较大的英雄 ID（恒有 `a < b`）。
    pub b: i64,
    /// `a` 对 `b` 的去偏残差（百分点，2 位小数，`0.0` = 五五开）。
    pub delta: f64,
    /// 全库对位有效样本（时间衰减 + BO3 系列降权后，四舍五入取整）。
    pub n: u64,
}

impl From<(i64, i64, f64, u64)> for GlobalMatchup {
    fn from((a, b, delta, n): (i64, i64, f64, u64)) -> Self {
        Self { a, b, delta, n }
    }
}

impl From<GlobalMatchup> for (i64, i64, f64, u64) {
    fn from(m: GlobalMatchup) -> Self {
        (m.a, m.b, m.delta, m.n)
    }
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
    fn matchup_serializes_expected_fields() {
        let m = Matchup {
            opponent: 16_000_001,
            delta: -12.95,
            samples: 37,
            eff_samples: 12.3333,
        };
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(
            json,
            r#"{"opponent":16000001,"delta":-12.95,"samples":37,"eff_samples":12.3333}"#
        );
    }

    #[test]
    fn global_matchup_serializes_as_compact_tuple() {
        // 数组格式：省掉 4 个键名，是控制 JSON 体积的关键
        let m = GlobalMatchup {
            a: 16_000_001,
            b: 16_000_002,
            delta: -1.25,
            n: 37,
        };
        assert_eq!(
            serde_json::to_string(&m).unwrap(),
            "[16000001,16000002,-1.25,37]"
        );
        // 反序列化闭环
        let back: GlobalMatchup = serde_json::from_str("[16000001,16000002,-1.25,37]").unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn tier_audit_detects_ineffective_filter() {
        let audit = TierAudit {
            min_tier: 16,
            battles_total: 100,
            battles_kept: 100,
            unknown_tier_battles: 100,
            distribution: vec![TierBucket {
                tier: 0,
                label: "未知".into(),
                battles: 100,
            }],
        };
        assert!(!audit.filter_effective(), "全为未知段位时门槛无从判定");
    }

    #[test]
    fn tier_audit_is_effective_when_tiers_are_known() {
        // 门槛在入库阶段生效，analyze 阶段没有可过滤的是正常情况
        let audit = TierAudit {
            min_tier: 16,
            battles_total: 105,
            battles_kept: 105,
            unknown_tier_battles: 0,
            distribution: vec![TierBucket {
                tier: 16,
                label: "传奇 I".into(),
                battles: 105,
            }],
        };
        assert!(audit.filter_effective());
    }

    #[test]
    fn tier_labels_follow_ranked_ladder() {
        assert_eq!(tier_label(0), "未知");
        assert_eq!(tier_label(1), "青铜 I");
        assert_eq!(tier_label(3), "青铜 III");
        assert_eq!(tier_label(10), "钻石 I");
        assert_eq!(tier_label(15), "神话 III");
        assert_eq!(tier_label(16), "传奇 I");
        assert_eq!(tier_label(18), "传奇 III");
        assert_eq!(tier_label(21), "大师 III");
        assert_eq!(tier_label(22), "职业");
        assert_eq!(tier_group(16), "传奇");
        assert_eq!(tier_group(22), "职业");
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
