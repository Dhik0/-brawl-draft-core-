//! 离线分析引擎：加权胜率、贝叶斯平滑、地图分层去偏的 Log5 克制残差与选人顺位。
//!
//! ## 三级聚合
//!
//! * **地图维度**（`maps[]`）：出场率、平滑胜率、综合评分、选人顺位，以及
//!   **地图级克制列表**。地形差异（草图 vs 开阔图）直接体现在这一层，
//!   是克制结论的首选来源。
//! * **模式维度**（`modes[]`）：同一模式的地图共享地形族特征，按模式汇总后
//!   样本量放大数倍，作为单图样本不足时的**先验**。
//! * **全局维度**（`global_matchups`）：全库 10 万+ 场对局汇总的 1v1 去偏残差，
//!   跨模式兜底。它**不按 `|delta|` 剪枝**，因此能明确回答「这一对是五五开」，
//!   与「这一对真的没有样本」区分开——前端做 3×3 对位展示时，前两层查不到
//!   就落到这一层，避免整片空白。
//!
//! ## 克制算法：地图分层去偏 + 互斥分层平滑
//!
//! 直接用「A 打 B 的原始胜率」当克制分是错的：强势英雄对谁都能赢，会被误判成
//! 「克制所有人」。正确做法是先扣掉双方**在该图上的基础强度**：
//!
//! ```text
//! p_a     = clamp((wins_a + m/2) / (games_a + m), 0.05, 0.95)   // 该图基础胜率
//! E_ab,m  = p_a(1-p_b) / (p_a(1-p_b) + (1-p_a)p_b)              // 该图 Log5 预期胜率
//! e_ab,m  = w_ab,m - n_ab,m * E_ab,m                            // 该图超额胜场
//! ```
//!
//! ### 为什么必须按图分层
//!
//! 如果先把各图的 `n` / `w` 加起来、再用「跨图混合的胜率」对照「跨图混合的
//! 预期」，就会踩辛普森悖论：两张图的出场结构不同（一张图对手偏弱、一张图
//! 对手偏强），混合后的残差会被地形差异污染，甚至把「只在开阔图强」的英雄
//! 判成「在草图被克制」。
//!
//! 本实现**每张图各自用自己的 `p_a` 算 `E`，再对超额胜场 `e` 求和**：`e` 已经
//! 扣掉了该图的基础强度，可以直接相加，这就是模式层 / 全库层的汇总口径。
//!
//! ### 互斥分层：本图 / 同模式其他图 / 其他模式
//!
//! 单图对位样本往往很薄，需要借外部样本做先验；但三组数据必须**互不重叠**，
//! 否则同一批对局会被重复计数：
//!
//! ```text
//! 本图         n1 = n(A,B,m)             e1 = e(A,B,m)
//! 同模式其他图  n2 = N_mode - n1          e2 = E_mode - e1
//! 其他模式      n3 = N_global - N_mode    e3 = E_global - E_mode
//! ```
//!
//! 外部样本按权重折算并封顶，避免其他模式的海量样本淹没本图的大样本：
//!
//! ```text
//! n2_eff = min(0.6 * n2, 40)   e2_eff = e2 * (n2_eff / n2)
//! n3_eff = min(0.3 * n3, 25)   e3_eff = e3 * (n3_eff / n3)
//! eff_n  = n1 + n2_eff + n3_eff
//! delta  = (e1 + e2_eff + e3_eff) / (eff_n + K)        // K = 20 伪场次
//! ```
//!
//! `delta` 是**概率残差**（`0.03` 表示 3 个百分点）。导出时乘以 100 记作
//! 百分点，与 v3 字段口径保持一致。
//!
//! 严格反对称：`E_ba = 1 - E_ab` 且 `e_ba = -e_ab`；三层的 `e` 都按规范方向
//! （`lo < hi`）算一次再镜像，因此 `delta(A,B,m) == -delta(B,A,m)` 精确成立。
//!
//! ### 过滤与兜底
//!
//! 三层各有自己的门槛（`K` 统一为 20 伪场次，不再用过强的收缩把中小样本压平）：
//!
//! ```text
//! 地图级  eff_n >= 8   且 |delta| >= 1.5pp
//! 模式级  eff_n >= 12  且 |delta| >= 1.5pp
//! 全局级  n     >= 10  不按 |delta| 剪枝（保留均势对位）
//! ```
//!
//! 地图级门槛最低，是因为单图对位天然稀薄；放宽到 `eff_n >= 8` 后，实测 3×3
//! 对位不再有大面积空白。全局级保留 `delta ≈ 0` 的条目，前端才能区分「五五开」
//! 和「无数据」。**后端不做任何「正残差过大就抹零」的处理**：真实克制关系原样
//! 保留，绿灯绝对胜率由前端展示层结合本图基础胜率自行兜底。
//!
//! ## 综合评分（`score`）
//!
//! 胜率主导、出场率对数压缩、低样本软衰减：
//!
//! ```text
//! rel      = games / (games + 40)                  // 可靠性
//! wr_core  = 0.5 + (p_smoothed - 0.5) * rel        // 薄样本拉回中性
//! use_term = ln(1 + share*k) / ln(1 + max_share*k) // 对数压缩，0~1
//! raw      = 0.92 * wr_core + 0.08 * use_term
//! ```
//!
//! `share` 是槽位占比（`use_rate / 6`）。Min-Max 的分母只统计**有效英雄**
//! （`games >= 10` 且 `share >= 0.5%`），极冷门英雄（`share < 0.3%`）的综合分
//! 封顶 35，不允许靠「平滑后恰好 50% 胜率」排进中上游。
//!
//! ## 顺位（`pick_order`）
//!
//! `flex` 的「被康特惩罚」只看**本图**的对位 delta，不用模式级结果：
//! 同一英雄在草图与开阔图上的克制差异很大，拿模式级最差对位去评一张开阔图，
//! 会被无关地形的近战克制污染。
//!
//! 只有 `games >= 20`、`share >= 0.5%` 且有效对位边 `>= 3` 的英雄参与
//! FIRST / LAST 分位划分；其余一律 `MIDDLE`——避免没有对位样本的冷门英雄
//! 因为克制方差为 0 而被误标成先手。
//!
//! ## 时间衰减
//!
//! 胜率榜每个样本按 `w = 0.5^(Δt / 30)` 加权（`Δt` 为距**最新对局**的天数），
//! 使旧版本平衡性调整的数据自然降权。
//!
//! ## BO3 系列降权
//!
//! 传奇排位是 BO3：同一批 6 名英雄在 10 分钟内会连打 2~3 小局。若把每个小局都当成
//! 独立样本，有效样本量会虚高 2~3 倍。因此在时间衰减之外再乘一个**系列权重**：
//!
//! ```text
//! 聚簇键   = (map, mode, canonical_teams)   // 两队各自升序，两队整体再按字典序
//! 时间窗口 = 相邻两局 Δt <= 600s            // 归入同一系列簇（链式合并）
//! 系列权重 = 1 / k                          // k = 簇内小局数
//! 最终权重 = 时间衰减 * 系列权重
//! ```
//!
//! `canonical_teams` 让交换红蓝边的小局也能归入同一簇；链式合并意味着 0s / 600s /
//! 1200s 三局同阵容对局属于同一个系列（每段间隔都没超过 10 分钟）。
//!
//! 一轮完整 BO3 因此只贡献 `3 * 1/3 = 1.0` 次独立 BP 权重。**地图展示的总场次
//! （`total_games`）仍保留原始对局数**，只有胜率、出场率、1v1 对位的有效样本量与
//! 有效胜场按降权口径计算；`Matchup.samples` 保留原始场次，`Matchup.eff_samples`
//! 是降权后的有效样本。
//!
//! ## 并行策略
//!
//! 先按地图分组（单线程 O(n)），再用 `rayon` 并行计算每张图的原始聚合与超额
//! 胜场；模式 / 全库的汇总必须串行（需要全部地图的结果），之后地图维度的
//! 导出再次并行。

use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use rayon::prelude::*;

use crate::db::{parse_battle_time, BattleRow};
use crate::models::output::{
    tier_label, AnalysisParams, BrawlerRef, GlobalMatchup, MapBrawlerStat, MapMeta, Matchup,
    MetaOutput,
    ModeBrawlerStat, ModeMeta, PickOrder, RolesConfig, TierAudit, TierBucket, SCHEMA_VERSION,
};

/// 综合评分中胜率项的权重（其余给对数压缩后的出场率）。
///
/// 0.92 是刻意偏向胜率的取值：出场率只用来在同一胜率档位内做区分，
/// 不允许「9% 出场 + 52% 胜率」的热门卡线性碾压「2.5% 出场 + 56% 胜率」的强卡。
const SCORE_WR_WEIGHT: f64 = 0.92;
/// 出场率对数压缩系数（作用在「每场出场数」上，见 [`use_share`]）。
const SCORE_USE_LOG_K: f64 = 300.0;
/// 低样本软衰减系数：`rel = games / (games + K)`，K 越大对薄样本越保守。
const SCORE_RELIABILITY_K: f64 = 40.0;
/// 参与评分分母（Min-Max 区间）的最小原始场次。
const SCORE_MIN_GAMES: u64 = 10;
/// 进入评分分母的最小槽位占比（0.5%）。
const SCORE_MIN_USE_SHARE: f64 = 0.005;
/// 低于该槽位占比（0.3%）视为极冷门，综合分封顶。
const SCORE_COLD_USE_SHARE: f64 = 0.003;
/// 极冷门英雄的综合分上限，禁止排进中上游。
const SCORE_COLD_CAP: f64 = 35.0;
/// 每场 3v3 共 6 个英雄槽位，`use_rate` 合计 ≈ 6。
const HERO_SLOTS_PER_BATTLE: f64 = 6.0;

/// 参与 FIRST / LAST 分位划分的最小原始场次。
const PICK_MIN_GAMES: u64 = 20;
/// 参与分位划分的最小槽位占比（0.5%）。
const PICK_MIN_USE_SHARE: f64 = 0.005;
/// 参与分位划分的最小有效对位边数。
const PICK_MIN_EDGES: usize = 3;

/// 分层贝叶斯收缩的先验强度 K（伪场次）。
///
/// 与旧版 `C = 15` 的差别：`delta` 现在已经是概率残差（未乘 100）。
/// `K = 20` 是「本图 20 场伪样本」的折中：既能压住 5 场 4 胜这类纯噪音，
/// 又不会把 20~40 场的中小样本真实优势压平（旧值 `K = 50` 会）。
pub const MATCHUP_PRIOR_K: f64 = 20.0;
/// 同模式其他地图的样本折算权重。
pub const SAME_MODE_SAMPLE_WEIGHT: f64 = 0.6;
/// 同模式其他地图折算后的样本上限（防止外部大样本压过本图）。
pub const SAME_MODE_SAMPLE_CAP: f64 = 40.0;
/// 其他模式的样本折算权重。
pub const OTHER_MODE_SAMPLE_WEIGHT: f64 = 0.3;
/// 其他模式折算后的样本上限。
pub const OTHER_MODE_SAMPLE_CAP: f64 = 25.0;
/// 地图级克制条目的最小有效样本 `eff_n`。
///
/// 单图对位天然稀薄，门槛比模式级更低——这是覆盖率的主要来源。
pub const MAP_MATCHUP_MIN_EFF_SAMPLES: f64 = 8.0;
/// 模式级克制条目的最小有效样本 `eff_n`。
///
/// 模式层样本比单图大数倍，门槛相应抬高，避免整张模式表被噪音灌满。
pub const MODE_MATCHUP_MIN_EFF_SAMPLES: f64 = 12.0;
/// 地图级 / 模式级克制条目的最小残差绝对值（概率单位，`0.015` = 1.5pp）。
pub const MIN_MATCHUP_ABS_DELTA: f64 = 0.015;
/// 全局兜底表的最小原始对位样本量。
///
/// 全局表**不按 `|delta|` 剪枝**：均势对位（`delta ≈ 0`）也要导出，前端才能
/// 把「五五开」和「无数据」区分开。
pub const GLOBAL_MATCHUP_MIN_SAMPLES: u64 = 10;
/// 全局兜底表的残差门槛，固定为 0（不剪枝）。
pub const GLOBAL_MATCHUP_MIN_ABS_DELTA: f64 = 0.0;

/// 时间衰减的默认半衰期（天）。
pub const DEFAULT_DECAY_HALF_LIFE_DAYS: f64 = 30.0;

/// BO3 系列聚簇的时间窗口（秒）。
///
/// 同一聚簇键内**相邻**两局的时间差不超过该值，就视为同一轮 BO3 的小局。
/// 600 秒 = 10 分钟，覆盖「同一批 6 人在一局结束后立刻重开」的实际节奏。
pub const SERIES_CLUSTER_WINDOW_SECS: i64 = 600;

/// 基础胜率的截断区间，防止 Log5 除零与极端值。
const BASE_RATE_FLOOR: f64 = 0.05;
const BASE_RATE_CEIL: f64 = 0.95;

/// 模式级克制导出条数上限（按 |delta| 取最显著的 K 条）。
const MODE_MATCHUP_TOP_K: usize = 12;
/// 地图级克制每侧（优势 / 劣势）导出条数上限。
const MAP_MATCHUP_TOP_K: usize = 8;

/// 每个地图上被判为 `FIRST` 的比例（分位分配，保证分布稳定）。
const FIRST_RATIO: f64 = 0.22;
/// 每个地图上被判为 `LAST` 的比例上限。
const LAST_RATIO: f64 = 0.22;
/// 对位 delta 极差低于该值（pp），视为「没有明显康特关系」，不判 `LAST`。
const MIN_POLARITY_PP: f64 = 8.0;
/// 存在明显劣势对位时的强度惩罚系数。
const FLEX_PENALTY: f64 = 0.30;
/// 最差对位 delta 高于该值（pp）不扣分。
const FLEX_PIVOT_PP: f64 = 4.0;
/// 单次扣分对应的 delta 跨度（pp）。
const FLEX_MAX_PENALTY_PP: f64 = 20.0;
/// 计算「波动」至少需要的对位条数，单条对位不构成波动。
const MIN_MATCHUPS_FOR_PROFILE: usize = 2;

/// 规范方向的对位键：`(lo, hi)` 且 `lo < hi`。
type PairKey = (i64, i64);
/// 模式层对位键：模式索引 + 规范方向的对位键。
type ModePairKey = (usize, i64, i64);

/// 分析参数。
#[derive(Debug, Clone)]
pub struct AnalyzeOptions {
    /// 胜率贝叶斯平滑的先验权重 m。
    pub bayes_prior_weight: f64,
    /// 分层收缩的先验强度 K（伪场次）。
    pub matchup_prior_k: f64,
    /// 段位阈值：仅保留 `avg_tier >= min_tier` 的对局（`avg_tier <= 0` 视为未知，保留）。
    pub min_tier: i64,
    /// 时间衰减半衰期（天）。测试可传 `f64::INFINITY` 关闭衰减。
    pub decay_half_life_days: f64,
    /// 赛季起始日期字符串（回显）。
    pub season_start: String,
    /// 英雄定位配置。
    pub roles: RolesConfig,
}

/// 对内存中的对局切片执行完整分析。
pub fn analyze(
    battles: &[BattleRow],
    brawler_dict: &HashMap<i64, String>,
    opts: &AnalyzeOptions,
) -> MetaOutput {
    // 0) 段位审计 + 过滤（不删库内原始数据，只在聚合口径上过滤）
    let tier_audit = build_tier_audit(battles, opts.min_tier);
    let kept: Vec<&BattleRow> = battles
        .iter()
        .filter(|b| tier_passes(b.avg_tier, opts.min_tier))
        .collect();

    let weighted = weigh_battles(&kept, opts.decay_half_life_days);

    // 1) 按地图分组（一次遍历，权重只算一次）
    let mut by_map: HashMap<&str, Vec<(&BattleRow, f64)>> = HashMap::new();
    for (row, weight) in &weighted {
        by_map
            .entry(row.map_name.as_str())
            .or_default()
            .push((*row, *weight));
    }

    // 2) 每张图各自的原始聚合 + 该图基础胜率 + 超额胜场（并行）
    let mut cores: Vec<MapCore> = by_map
        .par_iter()
        .map(|(map_name, rows)| build_map_core(map_name, rows, opts))
        .collect();
    cores.sort_by(|a, b| {
        b.total_games
            .cmp(&a.total_games)
            .then_with(|| a.map_name.cmp(&b.map_name))
    });

    // 3) 模式 / 全库的超额胜场汇总。
    //
    // 必须串行：这一步要看到全部地图的结果。汇总的是 `e`（已扣掉各图基础强度），
    // 不是重新用混合胜率算一次 Log5——见模块文档「为什么必须按图分层」。
    let mut mode_index: HashMap<&str, usize> = HashMap::new();
    let mut mode_names: Vec<&str> = Vec::new();
    for core in &cores {
        if !mode_index.contains_key(core.mode.as_str()) {
            mode_index.insert(core.mode.as_str(), mode_names.len());
            mode_names.push(core.mode.as_str());
        }
    }
    let (mode_excess, global_excess) = aggregate_excess(&cores, &mode_index);

    // 4) 地图维度并行导出（评分 + 顺位 + 地图级克制）
    let mut maps: Vec<MapMeta> = cores
        .par_iter()
        .map(|core| {
            let idx = mode_index[core.mode.as_str()];
            build_map(core, idx, &mode_excess, &global_excess, brawler_dict, opts)
        })
        .collect();
    maps.sort_by(|a, b| {
        b.total_games
            .cmp(&a.total_games)
            .then_with(|| a.map_name.cmp(&b.map_name))
    });

    // 5) 模式维度导出
    let mut modes: Vec<ModeMeta> = mode_names
        .iter()
        .enumerate()
        .map(|(idx, mode)| {
            let group: Vec<&MapCore> = cores
                .iter()
                .filter(|c| mode_index[c.mode.as_str()] == idx)
                .collect();
            build_mode(
                mode,
                idx,
                &group,
                &mode_excess,
                &global_excess,
                brawler_dict,
                opts,
            )
        })
        .collect();
    modes.sort_by(|a, b| {
        b.total_games
            .cmp(&a.total_games)
            .then_with(|| a.mode.cmp(&b.mode))
    });

    // 6) 全局 1v1 兜底表（第三层）：全库去偏残差，不按 |delta| 剪枝
    let global_matchups = build_global_matchups(&global_excess, opts.matchup_prior_k);

    MetaOutput {
        schema_version: SCHEMA_VERSION,
        updated_at: Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        season_start: opts.season_start.clone(),
        total_battles: kept.len() as u64,
        min_tier: opts.min_tier,
        tier_audit,
        params: AnalysisParams {
            bayes_prior_weight: opts.bayes_prior_weight,
            // 旧字段名沿用：语义已统一为分层收缩强度 K
            matchup_smoothing_c: opts.matchup_prior_k,
            // 三个旧字段都是「地图级门槛」的历史别名，保留以兼容 v3 消费方
            matchup_min_samples: MAP_MATCHUP_MIN_EFF_SAMPLES as u32,
            map_matchup_min_samples: MAP_MATCHUP_MIN_EFF_SAMPLES as u32,
            map_matchup_trust_samples: MAP_MATCHUP_MIN_EFF_SAMPLES as u32,
            decay_half_life_days: opts.decay_half_life_days,
            matchup_min_eff_samples: MAP_MATCHUP_MIN_EFF_SAMPLES,
            matchup_min_abs_delta: MIN_MATCHUP_ABS_DELTA,
            same_mode_weight: SAME_MODE_SAMPLE_WEIGHT,
            same_mode_cap: SAME_MODE_SAMPLE_CAP,
            other_mode_weight: OTHER_MODE_SAMPLE_WEIGHT,
            other_mode_cap: OTHER_MODE_SAMPLE_CAP,
            mode_matchup_min_eff_samples: MODE_MATCHUP_MIN_EFF_SAMPLES,
            global_matchup_min_samples: GLOBAL_MATCHUP_MIN_SAMPLES,
            global_matchup_min_abs_delta: GLOBAL_MATCHUP_MIN_ABS_DELTA,
        },
        brawlers: build_brawler_refs(&kept, brawler_dict),
        maps,
        modes,
        global_matchups,
        roles: resolve_roles(&opts.roles, brawler_dict),
    }
}

// ---------------------------------------------------------------------------
// 段位过滤
// ---------------------------------------------------------------------------

/// 段位是否达标。`avg_tier <= 0` 表示入库时该字段缺失，按「未知」处理并保留，
/// 否则会把整库数据静默清空——未知段位的数量由 `TierAudit` 单独报告。
fn tier_passes(avg_tier: i64, min_tier: i64) -> bool {
    avg_tier <= 0 || avg_tier >= min_tier
}

/// 统计段位分布并给出过滤前后的对局数。
pub fn build_tier_audit(battles: &[BattleRow], min_tier: i64) -> TierAudit {
    let mut dist: BTreeMap<i64, u64> = BTreeMap::new();
    let mut unknown = 0u64;
    for b in battles {
        *dist.entry(b.avg_tier).or_insert(0) += 1;
        if b.avg_tier <= 0 {
            unknown += 1;
        }
    }
    let kept = battles
        .iter()
        .filter(|b| tier_passes(b.avg_tier, min_tier))
        .count() as u64;

    TierAudit {
        min_tier,
        battles_total: battles.len() as u64,
        battles_kept: kept,
        unknown_tier_battles: unknown,
        distribution: dist
            .into_iter()
            .map(|(tier, battles)| TierBucket {
                tier,
                label: tier_label(tier),
                battles,
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// 时间衰减
// ---------------------------------------------------------------------------

/// 时间衰减权重：`w = 0.5^(Δt / half_life)`。
///
/// 最新对局 `Δt = 0` -> 权重 1.0；每往前推一个半衰期权重减半。
pub fn decay_weight(delta_days: f64, half_life_days: f64) -> f64 {
    if !half_life_days.is_finite() || half_life_days <= 0.0 {
        return 1.0;
    }
    0.5_f64.powf(delta_days.max(0.0) / half_life_days)
}

/// 给每条对局计算权重 = 时间衰减 × BO3 系列权重。基准是**数据集中最新的对局**，
/// 而不是 `Utc::now()`，这样离线重算历史数据时结果可复现。
fn weigh_battles<'a>(
    battles: &[&'a BattleRow],
    half_life_days: f64,
) -> Vec<(&'a BattleRow, f64)> {
    let times: Vec<Option<DateTime<Utc>>> = battles
        .iter()
        .map(|b| parse_battle_time(&b.battle_time))
        .collect();
    let latest = times.iter().flatten().copied().max();
    let series = series_weights(battles);

    battles
        .iter()
        .zip(times)
        .zip(series)
        .map(|((row, time), series_weight)| {
            let decay = match (latest, time) {
                (Some(latest), Some(t)) => {
                    let delta_days = (latest - t).num_seconds() as f64 / 86_400.0;
                    decay_weight(delta_days, half_life_days)
                }
                // 时间缺失时不做衰减，按满权重处理（入库前已校验过时间，属于兜底）
                _ => 1.0,
            };
            (*row, decay * series_weight)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// BO3 系列聚簇降权
// ---------------------------------------------------------------------------

/// 系列聚簇键：`(地图, 模式, 规范化阵容)`。
///
/// 规范化阵容 = 两队各自的英雄 ID 升序排列，再把两队整体按字典序排列——
/// 这样小局之间交换红蓝边也归入同一组。
type SeriesKey<'a> = (&'a str, &'a str, Vec<Vec<i64>>);

/// 把一局的两队英雄规范化成可比较的阵容：队内升序 + 队间字典序。
fn canonical_lineup(row: &BattleRow) -> Vec<Vec<i64>> {
    let mut teams = vec![row.winners.clone(), row.losers.clone()];
    for team in &mut teams {
        team.sort_unstable();
    }
    teams.sort();
    teams
}

/// 计算每局对局的**系列权重** `1/k`（`k` = 该局所属 BO3 系列簇内的小局数）。
///
/// 聚簇规则见模块文档「BO3 系列降权」：同 `(map, mode, canonical_teams)` 的对局按
/// 时间升序，相邻两局时间差 `<= 600s` 就并入同一簇（链式）。时间不可解析的对局
/// 无法判定先后，单独按 `1.0` 处理，绝不把不确定的数据合并降权。
///
/// 返回值的长度与 `battles` 一一对应；`sum(返回值)` 即「聚簇后的有效 BP 组数」。
pub fn series_weights<'a>(battles: &[&'a BattleRow]) -> Vec<f64> {
    let mut weights = vec![1.0_f64; battles.len()];
    let mut groups: HashMap<SeriesKey<'a>, Vec<(i64, usize)>> = HashMap::new();

    for (idx, row) in battles.iter().enumerate() {
        let Some(ts) = parse_battle_time(&row.battle_time) else {
            continue; // 时间不可解析：保持独立权重 1.0
        };
        groups
            .entry((row.map_name.as_str(), row.mode.as_str(), canonical_lineup(row)))
            .or_default()
            .push((ts.timestamp(), idx));
    }

    for mut entries in groups.into_values() {
        // 时间升序；同一秒内按原始索引排序，保证同一份数据每次聚簇结果一致
        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut start = 0usize;
        for i in 1..=entries.len() {
            let split = i == entries.len()
                || entries[i].0 - entries[i - 1].0 > SERIES_CLUSTER_WINDOW_SECS;
            if split {
                let k = (i - start) as f64;
                for &(_, idx) in &entries[start..i] {
                    weights[idx] = 1.0 / k;
                }
                start = i;
            }
        }
    }

    weights
}

/// 供 CLI 报告的聚簇摘要：`(原始对局数, 聚簇后有效 BP 组数)`。
///
/// 只统计通过段位门槛的对局，与 `analyze` 的口径保持一致。
pub fn series_cluster_report(battles: &[BattleRow], min_tier: i64) -> (u64, u64) {
    let kept: Vec<&BattleRow> = battles
        .iter()
        .filter(|b| tier_passes(b.avg_tier, min_tier))
        .collect();
    let groups: f64 = series_weights(&kept).iter().sum();
    (kept.len() as u64, groups.round() as u64)
}

// ---------------------------------------------------------------------------
// 累加器
// ---------------------------------------------------------------------------

/// 加权累加器：`games` / `wins` 是加权值，`raw_*` 是原始计数。
#[derive(Debug, Default, Clone, Copy)]
struct Accum {
    games: f64,
    wins: f64,
    raw_games: u64,
    raw_wins: u64,
}

impl Accum {
    fn add_appearance(&mut self, weight: f64, won: bool) {
        self.games += weight;
        self.raw_games += 1;
        if won {
            self.wins += weight;
            self.raw_wins += 1;
        }
    }

    /// 合并另一份累加结果（用于把多张地图合成一个模式）。
    fn merge(&mut self, other: &Accum) {
        self.games += other.games;
        self.wins += other.wins;
        self.raw_games += other.raw_games;
        self.raw_wins += other.raw_wins;
    }
}

/// 两两对位的计数。
///
/// `n` / `w` 是**有效计数**（已乘时间衰减与 BO3 系列权重），克制残差必须用它们，
/// 否则一轮 BO3 会被当成 3 次独立交锋；`raw_n` 保留原始交锋场次供展示。
#[derive(Debug, Default, Clone, Copy)]
struct PairStat {
    /// 有效交锋样本（加权）。
    n: f64,
    /// 前者对后者的有效胜场（加权）。
    w: f64,
    /// 原始交锋场次（未加权，展示用）。
    raw_n: u64,
}

impl PairStat {
    fn add(&mut self, weight: f64, won: bool) {
        self.n += weight;
        self.raw_n += 1;
        if won {
            self.w += weight;
        }
    }

    fn merge(&mut self, other: &PairStat) {
        self.n += other.n;
        self.w += other.w;
        self.raw_n += other.raw_n;
    }
}

/// 某一层（本图 / 同模式 / 全库）的超额胜场累计。
///
/// `e` 是**超额胜场**（真实胜场 - Log5 预期胜场），不是胜率差值：
/// 各图的地形结构不同，但 `e` 已经扣掉了各自的基础强度，可以直接相加。
#[derive(Debug, Default, Clone, Copy)]
struct Stratum {
    n: f64,
    e: f64,
}

/// 模式内某个英雄的对位画像（单位均为百分点）。
#[derive(Debug, Clone, Copy)]
struct MatchupProfile {
    /// 最差对位 delta（负数 = 被克制）。
    worst_pp: f64,
    /// 对位 delta 极差。
    spread_pp: f64,
}

/// 单张地图的中间聚合结果（模式 / 全库汇总与导出都要用）。
struct MapCore {
    map_name: String,
    /// 该图的主模式（同图多模式时取样本最多的一个）。
    mode: String,
    total_games: u64,
    total_weight: f64,
    acc: HashMap<i64, Accum>,
    /// 有向对位原始计数（双向各一条，保证 `w_ab + w_ba = n`）。
    pairs: HashMap<PairKey, PairStat>,
    /// 规范方向（`lo < hi`）的超额胜场，`e` 属于 `lo` 对 `hi`。
    excess: HashMap<PairKey, Stratum>,
}

// ---------------------------------------------------------------------------
// Log5 数学 + 分层平滑
// ---------------------------------------------------------------------------

/// 基础胜率截断，防止 Log5 除零与 0/1 极端值。
fn clamp_base(p: f64) -> f64 {
    p.clamp(BASE_RATE_FLOOR, BASE_RATE_CEIL)
}

/// Log5 预期对位胜率：`E_AB = p_A(1-p_B) / (p_A(1-p_B) + (1-p_A)p_B)`。
///
/// 满足 `E_BA = 1 - E_AB`，是克制残差严格反对称的基础。
pub fn log5(p_a: f64, p_b: f64) -> f64 {
    let a = clamp_base(p_a);
    let b = clamp_base(p_b);
    let num = a * (1.0 - b);
    let den = num + (1.0 - a) * b;
    if den <= f64::EPSILON {
        0.5
    } else {
        num / den
    }
}

/// 外部样本的折算：按权重缩小并封顶，`e` 按同一比例缩放。
///
/// 缩放比例作用在**超额胜场**上（而不是只改分母），这样外部样本的「证据强度」
/// 与它贡献的 `eff_n` 始终一致，不会出现「样本被砍掉一半但残差全额保留」。
fn discount_samples(n: f64, e: f64, weight: f64, cap: f64) -> (f64, f64) {
    if n <= 0.0 {
        return (0.0, 0.0);
    }
    let n_eff = (weight * n).min(cap);
    (n_eff, e * (n_eff / n))
}

/// 互斥分层贝叶斯平滑：把本图 / 同模式其他图 / 其他模式的超额胜场合并成一个残差。
///
/// * `n1` / `e1`：本图场次与超额胜场；
/// * `n2` / `e2`：同模式其他地图（**已扣除本图**，不重复计数）；
/// * `n3` / `e3`：其他模式（**已扣除本模式**，不重复计数）；
/// * `k`：先验强度（伪场次），由 [`AnalyzeOptions::matchup_prior_k`] 传入。
///
/// 返回 `(delta, eff_n)`：`delta` 是概率残差（未乘 100），`eff_n` 是折算后的
/// 有效样本量，用于导出过滤。
pub fn hierarchical_delta(
    n1: f64,
    e1: f64,
    n2: f64,
    e2: f64,
    n3: f64,
    e3: f64,
    k: f64,
) -> (f64, f64) {
    let (n2_eff, e2_eff) = discount_samples(n2, e2, SAME_MODE_SAMPLE_WEIGHT, SAME_MODE_SAMPLE_CAP);
    let (n3_eff, e3_eff) = discount_samples(n3, e3, OTHER_MODE_SAMPLE_WEIGHT, OTHER_MODE_SAMPLE_CAP);
    let eff_n = n1 + n2_eff + n3_eff;
    let delta = (e1 + e2_eff + e3_eff) / (eff_n + k.max(0.0));
    (delta, eff_n)
}

/// 导出门槛：有效样本达到该层门槛、且残差足够显著。
///
/// 地图级与模式级的 `min_eff` 不同（见模块文档），显著性门槛共用。
fn passes_export_filter(delta: f64, eff_n: f64, min_eff: f64) -> bool {
    eff_n >= min_eff && delta.abs() >= MIN_MATCHUP_ABS_DELTA
}

// ---------------------------------------------------------------------------
// 地图层的原始聚合
// ---------------------------------------------------------------------------

/// 单张地图的原始聚合：基础胜率、两两计数、逐图超额胜场。
fn build_map_core(map_name: &str, rows: &[(&BattleRow, f64)], opts: &AnalyzeOptions) -> MapCore {
    let total_weight: f64 = rows.iter().map(|(_, w)| *w).sum();
    let mut acc: HashMap<i64, Accum> = HashMap::new();
    let mut pairs: HashMap<PairKey, PairStat> = HashMap::new();
    let mut mode_votes: HashMap<&str, u64> = HashMap::new();

    for (row, weight) in rows {
        *mode_votes.entry(row.mode.as_str()).or_insert(0) += 1;
        for b in &row.winners {
            acc.entry(*b).or_default().add_appearance(*weight, true);
        }
        for b in &row.losers {
            acc.entry(*b).or_default().add_appearance(*weight, false);
        }
        for a in &row.winners {
            for b in &row.losers {
                if a == b {
                    continue; // 镜像对局不计入克制关系
                }
                pairs.entry((*a, *b)).or_default().add(*weight, true);
                pairs.entry((*b, *a)).or_default().add(*weight, false);
            }
        }
    }

    let mode = mode_votes
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))
        .map(|(m, _)| m.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // 该图的基础胜率 P(a, m)：平滑后钳制到 [0.05, 0.95]，供本图 Log5 使用。
    //
    // 用**加权**的 wins/games 而不是原始计数：`e = w - n * E` 里的 `w` / `n` 已经是
    // 加权值，基础胜率必须同口径，否则一轮 BO3 会同时抬高 `p` 和 `e`。
    let base: HashMap<i64, f64> = acc
        .iter()
        .map(|(id, a)| {
            let p = smoothed_rate(a.wins, a.games, opts.bayes_prior_weight);
            (*id, clamp_base(p))
        })
        .collect();

    // 逐图超额胜场：只算规范方向，镜像由导出阶段完成。
    let mut excess: HashMap<PairKey, Stratum> = HashMap::new();
    for ((a, b), p) in &pairs {
        if a >= b {
            continue;
        }
        let e_ab = log5(
            base.get(a).copied().unwrap_or(0.5),
            base.get(b).copied().unwrap_or(0.5),
        );
        let n = p.n;
        excess.insert(
            (*a, *b),
            Stratum {
                n,
                e: p.w - n * e_ab,
            },
        );
    }

    MapCore {
        map_name: map_name.to_string(),
        mode,
        total_games: rows.len() as u64,
        total_weight,
        acc,
        pairs,
        excess,
    }
}

/// 把各图的超额胜场汇总成「模式层」与「全库层」。
///
/// 汇总的是 `e`（已按各自图的基础胜率去偏），因此不存在辛普森悖论；
/// 同一批对局只会进入 `global` 与它所属的那个 `mode` 各一次。
fn aggregate_excess(
    cores: &[MapCore],
    mode_index: &HashMap<&str, usize>,
) -> (
    HashMap<ModePairKey, Stratum>,
    HashMap<PairKey, Stratum>,
) {
    let mut by_mode: HashMap<ModePairKey, Stratum> = HashMap::new();
    let mut global: HashMap<PairKey, Stratum> = HashMap::new();

    for core in cores {
        let mi = mode_index[core.mode.as_str()];
        for (pair, s) in &core.excess {
            let m = by_mode.entry((mi, pair.0, pair.1)).or_default();
            m.n += s.n;
            m.e += s.e;

            let g = global.entry(*pair).or_default();
            g.n += s.n;
            g.e += s.e;
        }
    }

    (by_mode, global)
}

// ---------------------------------------------------------------------------
// 克制残差（导出用，单位：百分点）
// ---------------------------------------------------------------------------

/// 单张地图的克制残差矩阵：`(A, B) -> delta_pp`，已过滤噪音并双向镜像。
fn map_deltas(
    core: &MapCore,
    mode_idx: usize,
    mode_excess: &HashMap<ModePairKey, Stratum>,
    global_excess: &HashMap<PairKey, Stratum>,
    k: f64,
) -> HashMap<PairKey, f64> {
    let mut out: HashMap<PairKey, f64> = HashMap::with_capacity(core.excess.len() * 2);

    for ((lo, hi), s1) in &core.excess {
        let m = mode_excess.get(&(mode_idx, *lo, *hi)).copied().unwrap_or(*s1);
        let g = global_excess.get(&(*lo, *hi)).copied().unwrap_or(*s1);

        // 三组互斥数据：本图 / 同模式其他图 / 其他模式
        let (delta, eff_n) = hierarchical_delta(
            s1.n,
            s1.e,
            (m.n - s1.n).max(0.0),
            m.e - s1.e,
            (g.n - m.n).max(0.0),
            g.e - m.e,
            k,
        );
        if !passes_export_filter(delta, eff_n, MAP_MATCHUP_MIN_EFF_SAMPLES) {
            continue;
        }
        let pp = round2(delta * 100.0);
        out.insert((*lo, *hi), pp);
        // 精确镜像，保证 delta_BA == -delta_AB
        out.insert((*hi, *lo), -pp);
    }

    out
}

/// 模式层的克制残差：把「本图」换成「整个模式」，其他模式作为外部先验。
fn mode_deltas(
    mode_idx: usize,
    mode_excess: &HashMap<ModePairKey, Stratum>,
    global_excess: &HashMap<PairKey, Stratum>,
    k: f64,
) -> HashMap<PairKey, f64> {
    let mut out: HashMap<PairKey, f64> = HashMap::new();

    for ((mi, lo, hi), s) in mode_excess {
        if *mi != mode_idx {
            continue;
        }
        let g = global_excess.get(&(*lo, *hi)).copied().unwrap_or(*s);
        let (delta, eff_n) = hierarchical_delta(
            s.n,
            s.e,
            0.0,
            0.0,
            (g.n - s.n).max(0.0),
            g.e - s.e,
            k,
        );
        if !passes_export_filter(delta, eff_n, MODE_MATCHUP_MIN_EFF_SAMPLES) {
            continue;
        }
        let pp = round2(delta * 100.0);
        out.insert((*lo, *hi), pp);
        out.insert((*hi, *lo), -pp);
    }

    out
}

/// 全局 1v1 兜底表（第三层）：全库去偏残差，**不按 `|delta|` 剪枝**。
///
/// * 只用规范方向 `lo < hi`，每对英雄一条记录；
/// * 门槛只有样本 `n >= GLOBAL_MATCHUP_MIN_SAMPLES`（`n` 是 BO3 降权后的有效样本），
///   `delta ≈ 0` 的均势对位也会保留——这正是前端区分「五五开」与「无数据」的依据；
/// * 收缩口径与地图/模式层一致：`delta = Σe / (n + K)`，保证三层同尺度；
/// * 输出按 `(lo, hi)` 排序，保证导出文件稳定可 diff。
fn build_global_matchups(
    global_excess: &HashMap<PairKey, Stratum>,
    k: f64,
) -> Vec<GlobalMatchup> {
    let mut out: Vec<GlobalMatchup> = global_excess
        .iter()
        .filter_map(|((lo, hi), s)| {
            let n = s.n.round() as u64;
            if n < GLOBAL_MATCHUP_MIN_SAMPLES {
                return None;
            }
            let delta_pp = round2(s.e / (s.n + k.max(0.0)) * 100.0);
            if delta_pp.abs() < GLOBAL_MATCHUP_MIN_ABS_DELTA {
                return None;
            }
            Some(GlobalMatchup {
                a: *lo,
                b: *hi,
                delta: delta_pp,
                n,
            })
        })
        .collect();
    out.sort_by_key(|m| (m.a, m.b));
    out
}

/// 从一组对位统计中提取每个英雄的对位画像（最差 delta + 极差）。
///
/// 只统计**通过导出门槛**的对位边；不足 [`MIN_MATCHUPS_FOR_PROFILE`] 条边的英雄
/// 不产生画像（没有画像就不会被判 `LAST`，也不会被「被康特」惩罚扣分）。
fn extract_profiles(
    pairs: &HashMap<PairKey, PairStat>,
    delta: &HashMap<PairKey, f64>,
) -> HashMap<i64, MatchupProfile> {
    let mut agg: HashMap<i64, (f64, f64, usize)> = HashMap::new();
    for (from, to) in pairs.keys() {
        let Some(&d) = delta.get(&(*from, *to)) else {
            continue;
        };
        let e = agg
            .entry(*from)
            .or_insert((f64::INFINITY, f64::NEG_INFINITY, 0));
        e.0 = e.0.min(d);
        e.1 = e.1.max(d);
        e.2 += 1;
    }

    agg.into_iter()
        .filter(|(_, (_, _, count))| *count >= MIN_MATCHUPS_FOR_PROFILE)
        .map(|(id, (worst, best, _))| {
            (
                id,
                MatchupProfile {
                    worst_pp: worst,
                    spread_pp: best - worst,
                },
            )
        })
        .collect()
}

/// 按英雄枚举已导出的克制条目（只保留 `delta` 里存在的边）。
///
/// `samples` 是原始交锋场次，`eff_samples` 是应用 BO3 系列降权后的有效样本——
/// 一轮 3 局的 BO3 在这里只算 1 场。
fn matchups_of(
    id: i64,
    pairs: &HashMap<PairKey, PairStat>,
    delta: &HashMap<PairKey, f64>,
) -> Vec<Matchup> {
    pairs
        .iter()
        .filter(|((from, _), _)| *from == id)
        .filter_map(|((_, to), p)| {
            delta.get(&(id, *to)).map(|d| Matchup {
                opponent: *to,
                delta: *d,
                samples: p.raw_n,
                eff_samples: round4(p.n),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 模式维度
// ---------------------------------------------------------------------------

fn build_mode(
    mode: &str,
    mode_idx: usize,
    cores: &[&MapCore],
    mode_excess: &HashMap<ModePairKey, Stratum>,
    global_excess: &HashMap<PairKey, Stratum>,
    dict: &HashMap<i64, String>,
    opts: &AnalyzeOptions,
) -> ModeMeta {
    // 每场对局只属于一张图，所以模式层的累加就是各图累加之和
    let mut acc: HashMap<i64, Accum> = HashMap::new();
    let mut pairs: HashMap<PairKey, PairStat> = HashMap::new();
    let mut total_games = 0u64;
    let mut total_weight = 0.0f64;

    for core in cores {
        total_games += core.total_games;
        total_weight += core.total_weight;
        for (id, a) in &core.acc {
            acc.entry(*id).or_default().merge(a);
        }
        for (key, p) in &core.pairs {
            pairs.entry(*key).or_default().merge(p);
        }
    }

    let delta = mode_deltas(mode_idx, mode_excess, global_excess, opts.matchup_prior_k);

    let mut brawlers: Vec<ModeBrawlerStat> = acc
        .iter()
        .map(|(id, a)| {
            let mut matchups = matchups_of(*id, &pairs, &delta);
            // 先按显著度取 Top K，再按强度排序，兼顾体积与可读性
            matchups.sort_by(|x, y| {
                y.delta
                    .abs()
                    .partial_cmp(&x.delta.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| x.opponent.cmp(&y.opponent))
            });
            matchups.truncate(MODE_MATCHUP_TOP_K);
            sort_matchups_by_delta(&mut matchups);

            ModeBrawlerStat {
                id: *id,
                name: name_of(*id, dict),
                games: a.raw_games,
                wins: a.raw_wins,
                win_rate: round4(ratio(a.wins, a.games)),
                smoothed_wr: round4(smoothed_rate(a.wins, a.games, opts.bayes_prior_weight)),
                use_rate: round4(ratio(a.games, total_weight)),
                matchups,
            }
        })
        .collect();

    brawlers.sort_by(|a, b| {
        b.smoothed_wr
            .partial_cmp(&a.smoothed_wr)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.games.cmp(&a.games))
            .then_with(|| a.id.cmp(&b.id))
    });

    ModeMeta {
        mode: mode.to_string(),
        total_games,
        brawlers,
    }
}

// ---------------------------------------------------------------------------
// 地图维度
// ---------------------------------------------------------------------------

/// 参与顺位分配的地图英雄条目（`flex` 与 `spread` 只在分配阶段使用）。
struct DraftCandidate {
    stat: MapBrawlerStat,
    /// 综合强度：平滑胜率减去「被康特」惩罚。
    flex: f64,
    /// 本图对位 delta 极差（pp）。
    spread_pp: f64,
    /// 是否有资格参与 FIRST / LAST 分位划分（样本、出场率、对位边数都要够）。
    eligible: bool,
}

fn build_map(
    core: &MapCore,
    mode_idx: usize,
    mode_excess: &HashMap<ModePairKey, Stratum>,
    global_excess: &HashMap<PairKey, Stratum>,
    dict: &HashMap<i64, String>,
    opts: &AnalyzeOptions,
) -> MapMeta {
    // 本图 delta：本图残差为主，同模式 / 其他模式按互斥权重做先验
    let delta = map_deltas(core, mode_idx, mode_excess, global_excess, opts.matchup_prior_k);

    // 本图对位画像：顺位判定只看这张图上已导出的克制关系
    let map_profiles = extract_profiles(&core.pairs, &delta);

    let max_use = core
        .acc
        .values()
        .map(|a| ratio(a.games, core.total_weight))
        .fold(0.0_f64, f64::max);

    // 第一轮：算平滑胜率、出场率与未归一化的综合分
    let mut pre: Vec<(i64, Accum, f64, f64)> = core
        .acc
        .iter()
        .map(|(id, a)| {
            let use_rate = ratio(a.games, core.total_weight);
            let smoothed = smoothed_rate(a.wins, a.games, opts.bayes_prior_weight);
            let raw_score = raw_score_of(a.raw_games, use_rate, smoothed, max_use);
            (*id, *a, smoothed, raw_score)
        })
        .collect();

    // 第二轮：Min-Max 缩放到 0~100。
    //
    // 区间只用**有效英雄**（场次够 + 出场率够）计算：极冷门英雄的原始分
    // 本来就贴近下限，把它们算进分母会把整个有效区间压扁，制造断崖。
    let (raw_min, raw_max) = pre
        .iter()
        .filter(|(_, a, _, _)| is_score_eligible(a.raw_games, ratio(a.games, core.total_weight)))
        .map(|p| p.3)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    let span = raw_max - raw_min;
    let degenerate = !span.is_finite() || span <= 1e-12;

    let mut candidates: Vec<DraftCandidate> = pre
        .drain(..)
        .map(|(id, a, smoothed, raw_score)| {
            let use_rate = ratio(a.games, core.total_weight);
            let score = normalize_score(raw_score, raw_min, span, degenerate);
            // 极冷门英雄（场次或出场率不足）不给高分，直接封顶
            let score = if is_cold(a.raw_games, use_rate) {
                score.min(SCORE_COLD_CAP)
            } else {
                score
            };

            let (worst_pp, spread_pp) = match map_profiles.get(&id) {
                Some(p) => (Some(p.worst_pp), p.spread_pp),
                None => (None, 0.0),
            };
            // 存在明显劣势对位时扣分：delta 越低扣得越多。
            // 这里的 delta 全部来自**本图**，不会把别的地形的克制算进来。
            let penalty = worst_pp
                .map(|w| ((FLEX_PIVOT_PP - w) / FLEX_MAX_PENALTY_PP).clamp(0.0, 1.0))
                .unwrap_or(0.0);
            let flex = smoothed - FLEX_PENALTY * penalty;

            let mut matchups = matchups_of(id, &core.pairs, &delta);
            let edge_count = matchups.len();
            truncate_matchups(&mut matchups);

            DraftCandidate {
                stat: MapBrawlerStat {
                    id,
                    name: name_of(id, dict),
                    games: a.raw_games,
                    wins: a.raw_wins,
                    win_rate: round4(ratio(a.wins, a.games)),
                    smoothed_wr: round4(smoothed),
                    use_rate: round4(use_rate),
                    score: round4(score),
                    pick_order: PickOrder::Middle,
                    matchups,
                },
                flex,
                spread_pp,
                eligible: is_pick_eligible(a.raw_games, use_rate, edge_count),
            }
        })
        .collect();

    assign_pick_orders(&mut candidates);

    let mut brawlers: Vec<MapBrawlerStat> = candidates.into_iter().map(|c| c.stat).collect();
    brawlers.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.games.cmp(&a.games))
            .then_with(|| a.id.cmp(&b.id))
    });

    MapMeta {
        map_name: core.map_name.clone(),
        mode: core.mode.clone(),
        total_games: core.total_games,
        brawlers,
    }
}

/// 按分位分配顺位，保证分布稳定，而不是被固定阈值卡成一边倒。
///
/// **只有 `eligible` 的英雄参与划分**：本图出场够多、且有足够对位边的英雄。
/// 极冷门英雄（如狙击图上的下水道近战）几乎没有对位样本，克制方差会被算成 0，
/// 旧逻辑会把它们全部误标成 `FIRST`——现在一律留在 `MIDDLE`。
///
/// 1. `LAST`：合格英雄中对位极差最大的那一批（需达到 `MIN_POLARITY_PP`）；
/// 2. `FIRST`：合格且未判 `LAST` 的英雄里 `flex` 最高的一批；
/// 3. 其余（含全部不合格者）`MIDDLE`。
///
/// 先判 `LAST` 再判 `FIRST`：波动极大的英雄本质上不适合先手。
fn assign_pick_orders(candidates: &mut [DraftCandidate]) {
    let n = candidates.len();
    if n == 0 {
        return;
    }
    let pool: Vec<usize> = (0..n).filter(|&i| candidates[i].eligible).collect();
    if pool.is_empty() {
        return; // 全员样本不足：不给任何顺位标签
    }
    let m = pool.len();
    let quota_first = (((m as f64) * FIRST_RATIO).round() as usize).clamp(1, m);
    let quota_last = (((m as f64) * LAST_RATIO).round() as usize).clamp(1, m);

    // 1) LAST：波动大且确有康特关系
    let mut by_spread = pool.clone();
    by_spread.sort_by(|&a, &b| {
        candidates[b]
            .spread_pp
            .partial_cmp(&candidates[a].spread_pp)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| candidates[a].stat.id.cmp(&candidates[b].stat.id))
    });
    let mut last_taken = 0usize;
    for &i in &by_spread {
        if last_taken >= quota_last {
            break;
        }
        if candidates[i].spread_pp >= MIN_POLARITY_PP {
            candidates[i].stat.pick_order = PickOrder::Last;
            last_taken += 1;
        }
    }

    // 2) FIRST：合格且剩余的英雄里强度最高的
    let mut rest: Vec<usize> = pool
        .into_iter()
        .filter(|&i| candidates[i].stat.pick_order == PickOrder::Middle)
        .collect();
    rest.sort_by(|&a, &b| {
        candidates[b]
            .flex
            .partial_cmp(&candidates[a].flex)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| candidates[a].stat.id.cmp(&candidates[b].stat.id))
    });
    for &i in rest.iter().take(quota_first) {
        candidates[i].stat.pick_order = PickOrder::First;
    }
}

// ---------------------------------------------------------------------------
// 综合评分
// ---------------------------------------------------------------------------

/// 出场率换算成「槽位占比」（全部英雄合计 ≈ 1）。
///
/// 导出的 `use_rate` 是**每场出场数**（合计 ≈ 6），换算成占比需除以 6。
/// 阈值按占比表述（0.5% / 0.3%），与直觉一致。
fn use_share(use_rate: f64) -> f64 {
    use_rate / HERO_SLOTS_PER_BATTLE
}

/// 未归一化的综合分：胜率主导 + 对数压缩的出场率。
///
/// * `wr_core = 0.5 + (p - 0.5) * rel`，`rel = games/(games+K)`：
///   0~5 场的冷门英雄即便平滑后停在 50%，也会被可靠性系数拉回中性，
///   不会白嫖 50% 胜率排到正常英雄前面；
/// * `use_term` 用 `ln(1 + share*K)` 再除以榜首值做 0~1 归一，
///   把 0.5% 与 10% 的差距从 20 倍压到 3 倍左右。
fn raw_score_of(games: u64, use_rate: f64, smoothed_wr: f64, max_use_rate: f64) -> f64 {
    let g = games as f64;
    let rel = g / (g + SCORE_RELIABILITY_K);
    let wr_core = 0.5 + (smoothed_wr - 0.5) * rel;

    let denom = (1.0 + use_share(max_use_rate) * SCORE_USE_LOG_K).ln();
    let use_term = if denom <= f64::EPSILON {
        0.0
    } else {
        ((1.0 + use_share(use_rate) * SCORE_USE_LOG_K).ln() / denom).clamp(0.0, 1.0)
    };

    SCORE_WR_WEIGHT * wr_core + (1.0 - SCORE_WR_WEIGHT) * use_term
}

/// 是否进入 Min-Max 分母（样本与出场率都要够）。
fn is_score_eligible(games: u64, use_rate: f64) -> bool {
    games >= SCORE_MIN_GAMES && use_share(use_rate) >= SCORE_MIN_USE_SHARE
}

/// 是否属于极冷门（综合分必须封顶）。
///
/// 门槛比 [`is_score_eligible`] 更宽松：0.3%~0.5% 槽位的英雄仍参与归一化，
/// 只是不给它进分母；低于 0.3% 或不足 10 场的直接压到低分段。
fn is_cold(games: u64, use_rate: f64) -> bool {
    games < SCORE_MIN_GAMES || use_share(use_rate) < SCORE_COLD_USE_SHARE
}

/// 归一化到 0~100，并对极冷门英雄封顶。
fn normalize_score(raw: f64, raw_min: f64, span: f64, degenerate: bool) -> f64 {
    if degenerate {
        // 只有一名英雄，或所有英雄指标完全一致：给中性分，避免虚高到 100
        return 50.0;
    }
    let score = (raw - raw_min) / span * 100.0;
    if !score.is_finite() {
        return 0.0;
    }
    score.clamp(0.0, 100.0)
}

/// 是否有资格参与 FIRST / LAST 分位划分。
fn is_pick_eligible(games: u64, use_rate: f64, edges: usize) -> bool {
    games >= PICK_MIN_GAMES
        && use_share(use_rate) >= PICK_MIN_USE_SHARE
        && edges >= PICK_MIN_EDGES
}

/// 导出的地图条目是否有资格参与 FIRST / LAST 分位划分。
///
/// 供 CLI 统计与前端展示复用，避免各处重复实现门槛判断。
/// 注意 `matchups` 已被截断到每侧 8 条，但门槛只要求 3 条，截断不影响判定。
pub fn is_ranked_pick_candidate(stat: &MapBrawlerStat) -> bool {
    is_pick_eligible(stat.games, stat.use_rate, stat.matchups.len())
}

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------

/// 按 delta 降序排列，同值时按对手 ID 升序（保证输出稳定）。
fn sort_matchups_by_delta(matchups: &mut [Matchup]) {
    matchups.sort_by(|x, y| {
        y.delta
            .partial_cmp(&x.delta)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.opponent.cmp(&y.opponent))
    });
}

/// 只保留「前 K 个优势 + 前 K 个劣势」，再按 delta 降序输出。
///
/// 正负分开取，避免全图都是优势对位时把「劣势」位置浪费在中性对位上。
fn truncate_matchups(matchups: &mut Vec<Matchup>) {
    sort_matchups_by_delta(matchups);
    if matchups.len() <= MAP_MATCHUP_TOP_K * 2 {
        return;
    }
    let split = matchups.partition_point(|m| m.delta > 0.0);
    let mut kept: Vec<Matchup> = Vec::with_capacity(MAP_MATCHUP_TOP_K * 2);
    // 优势：delta 最大的前 K 个
    kept.extend(matchups[..split.min(MAP_MATCHUP_TOP_K)].iter().cloned());
    // 劣势：delta 最小的后 K 个（列表尾部）
    let tail_start = matchups.len().saturating_sub(MAP_MATCHUP_TOP_K).max(split);
    kept.extend(matchups[tail_start..].iter().cloned());
    sort_matchups_by_delta(&mut kept);
    *matchups = kept;
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > f64::EPSILON {
        numerator / denominator
    } else {
        0.0
    }
}

/// 贝叶斯平滑：`(wins + m * 0.5) / (games + m)`，先验胜率固定 50%。
fn smoothed_rate(wins: f64, games: f64, prior_weight: f64) -> f64 {
    let m = prior_weight.max(0.0);
    if games + m <= f64::EPSILON {
        0.5
    } else {
        (wins + m * 0.5) / (games + m)
    }
}

fn name_of(id: i64, dict: &HashMap<i64, String>) -> String {
    dict.get(&id)
        .cloned()
        .unwrap_or_else(|| format!("Unknown#{id}"))
}

/// 英雄字典：优先使用对局中出现过的英雄，按 ID 升序。
fn build_brawler_refs(battles: &[&BattleRow], dict: &HashMap<i64, String>) -> Vec<BrawlerRef> {
    let mut seen: Vec<i64> = Vec::new();
    for row in battles {
        for id in row.winners.iter().chain(row.losers.iter()) {
            if !seen.contains(id) {
                seen.push(*id);
            }
        }
    }
    seen.sort_unstable();
    seen.into_iter()
        .map(|id| BrawlerRef {
            id,
            name: name_of(id, dict),
        })
        .collect()
}

/// 把定位配置里的英雄名解析成 ID（配置用名字而非硬编码 ID，避免与官方 ID 漂移）。
fn resolve_roles(roles: &RolesConfig, dict: &HashMap<i64, String>) -> RolesConfig {
    let by_name: HashMap<&str, i64> = dict.iter().map(|(id, name)| (name.as_str(), *id)).collect();
    let mut out = roles.clone();
    for role in &mut out.roles {
        role.brawler_ids = role
            .brawlers
            .iter()
            .filter_map(|name| by_name.get(name.trim()).copied())
            .collect();
    }
    out
}

/// 保留 4 位小数，压缩 JSON 体积。
fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

/// 保留 2 位小数（克制 delta 的导出精度），并把 `-0.0` 归一成 `0.0`。
fn round2(v: f64) -> f64 {
    if !v.is_finite() {
        return 0.0;
    }
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

    fn row_at(map: &str, mode: &str, time: &str, winners: [i64; 3], losers: [i64; 3]) -> BattleRow {
        BattleRow {
            battle_hash: format!("{map}-{mode}-{}-{time}", winners[0]),
            battle_time: time.into(),
            mode: mode.into(),
            map_name: map.into(),
            avg_tier: 25,
            winners: winners.to_vec(),
            losers: losers.to_vec(),
        }
    }

    fn row(map: &str, mode: &str, winners: [i64; 3], losers: [i64; 3]) -> BattleRow {
        row_at(map, mode, "20261020T143015.000Z", winners, losers)
    }

    fn opts() -> AnalyzeOptions {
        AnalyzeOptions {
            bayes_prior_weight: 50.0,
            matchup_prior_k: MATCHUP_PRIOR_K,
            min_tier: 20,
            // 单元测试默认关闭时间衰减，断言才能按「每场权重 1.0」手算复核。
            // 衰减本身由 `win_rate_is_time_decay_weighted` 单独覆盖。
            decay_half_life_days: f64::INFINITY,
            season_start: "2026-07-20".into(),
            roles: RolesConfig {
                version: 1,
                roles: vec![],
            },
        }
    }

    /// 测试基准时间。
    const TEST_BASE_TIME: &str = "20261020T000000.000Z";
    /// 测试用的相邻对局步长：> 600 秒，保证不会被 BO3 聚簇误合并。
    const TEST_STEP_SECS: i64 = 601;

    /// 以 `TEST_BASE_TIME` 为基准偏移若干秒，生成 `battleTime` 字符串。
    fn shift_time(offset_secs: i64) -> String {
        let base = parse_battle_time(TEST_BASE_TIME).expect("基准时间必须可解析");
        crate::db::format_battle_time(&(base + chrono::Duration::seconds(offset_secs)))
    }

    fn dict(ids: &[i64]) -> HashMap<i64, String> {
        ids.iter().map(|id| (*id, format!("B{id}"))).collect()
    }

    fn find_mode<'a>(out: &'a MetaOutput, mode: &str) -> &'a ModeMeta {
        out.modes
            .iter()
            .find(|m| m.mode == mode)
            .expect("模式应存在")
    }

    fn find_map<'a>(out: &'a MetaOutput, name: &str) -> &'a MapMeta {
        out.maps
            .iter()
            .find(|m| m.map_name == name)
            .expect("地图应存在")
    }

    fn find_mode_brawler(mode: &ModeMeta, id: i64) -> &ModeBrawlerStat {
        mode.brawlers
            .iter()
            .find(|b| b.id == id)
            .expect("英雄应存在")
    }

    /// 取某英雄在某图上的某条克制记录（不存在返回 None）。
    fn matchup_on_map(map: &MapMeta, from: i64, to: i64) -> Option<Matchup> {
        map.brawlers
            .iter()
            .find(|b| b.id == from)
            .and_then(|b| b.matchups.iter().find(|m| m.opponent == to))
            .cloned()
    }

    fn dummy_candidate(id: i64, flex: f64, spread_pp: f64) -> DraftCandidate {
        DraftCandidate {
            stat: MapBrawlerStat {
                id,
                name: format!("B{id}"),
                games: 40,
                wins: 20,
                win_rate: 0.5,
                smoothed_wr: 0.5,
                use_rate: 0.1,
                score: 50.0,
                pick_order: PickOrder::Middle,
                matchups: vec![],
            },
            flex,
            spread_pp,
            eligible: true,
        }
    }

    /// 构造 n 场 `winners` 击败 `losers` 的对局。
    ///
    /// 时间按 `TEST_STEP_SECS` 递增：这 n 场阵容完全相同，若时间戳相同会被 BO3
    /// 聚簇合并成一个系列（每场权重 1/n），那样断言就不再是「n 场独立样本」。
    fn battles_with(
        map: &str,
        mode: &str,
        winners: [i64; 3],
        losers: [i64; 3],
        n: usize,
    ) -> Vec<BattleRow> {
        (0..n)
            .map(|i| {
                let mut r = row_at(
                    map,
                    mode,
                    &shift_time(i as i64 * TEST_STEP_SECS),
                    winners,
                    losers,
                );
                r.battle_hash = format!("{map}-{i}");
                r
            })
            .collect()
    }

    /// 对位阶梯：n 场 `A(1)` 击败 `B(2)`，外加 n 场 `C(3)` 击败 `A(1)`。
    ///
    /// A 的总战绩是 n 胜 n 负，基础胜率恰好 50%（平滑后也精确为 0.5），
    /// 于是 A 对 B 的超额胜场完全来自「对位关系」本身，可以手算复核：
    ///
    /// ```text
    /// p_B   = 25 / (n + 50)              // B 全败
    /// E_AB  = 1 - p_B
    /// e     = n * (1 - E_AB) = 25n / (n + 50)
    /// delta = e / (n + K) = 25n / ((n + 50)(n + 20))      // K = 20
    /// ```
    ///
    /// n = 10 -> 13.89pp，n = 25 -> 18.52pp，n = 50 -> 17.86pp。
    fn counter_ladder(map: &str, mode: &str, n: usize, tag: &str) -> Vec<BattleRow> {
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            // 同一层的 n 场阵容相同，时间必须错开，避免被 BO3 聚簇合并
            let t = shift_time(i as i64 * TEST_STEP_SECS);

            let mut win = row_at(map, mode, &t, [1, 10, 11], [2, 12, 13]);
            win.battle_hash = format!("{tag}-w{i}");
            out.push(win);

            // 输的那局是另一套阵容（3 号击败 1 号），与上面不构成同一系列
            let mut lose = row_at(map, mode, &t, [3, 12, 13], [1, 10, 11]);
            lose.battle_hash = format!("{tag}-l{i}");
            out.push(lose);
        }
        out
    }

    const ALL: [i64; 6] = [1, 2, 3, 10, 11, 12];

    // -----------------------------------------------------------------------
    // Log5 数学
    // -----------------------------------------------------------------------

    #[test]
    fn log5_is_complementary() {
        for (a, b) in [(0.6, 0.4), (0.5, 0.5), (0.95, 0.05), (0.3, 0.8)] {
            let e_ab = log5(a, b);
            let e_ba = log5(b, a);
            assert!(
                (e_ab + e_ba - 1.0).abs() < 1e-12,
                "E_AB + E_BA 必须为 1，实际 {e_ab} + {e_ba}"
            );
        }
    }

    #[test]
    fn log5_equal_strength_is_even() {
        assert!((log5(0.5, 0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn log5_clamps_extreme_base_rates() {
        // p=1.0 与 p=0.0 会被截断到 0.95 / 0.05，不能出现除零或 0/1 胜率
        let e = log5(1.0, 0.0);
        assert!(e.is_finite() && (0.0..=1.0).contains(&e));
        assert!((e - 0.95 * 0.95 / (0.95 * 0.95 + 0.05 * 0.05)).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 互斥分层平滑
    // -----------------------------------------------------------------------

    #[test]
    fn hierarchical_delta_is_antisymmetric() {
        // 交换 A/B 时：n1/n2/n3 不变，e1/e2/e3 全部取反 -> delta 精确取反
        let (ab, eff_ab) = hierarchical_delta(40.0, 6.0, 100.0, 12.0, 200.0, -8.0, MATCHUP_PRIOR_K);
        let (ba, eff_ba) =
            hierarchical_delta(40.0, -6.0, 100.0, -12.0, 200.0, 8.0, MATCHUP_PRIOR_K);
        assert!((ab + ba).abs() < 1e-12, "delta_BA 必须是 -delta_AB");
        assert!((eff_ab - eff_ba).abs() < 1e-12);
    }

    #[test]
    fn prior_strength_is_twenty_pseudo_games() {
        // K 从 50 下调到 20：同样 10 场 6 胜（e = 1），残差从 1/60 放大到 1/30
        assert_eq!(MATCHUP_PRIOR_K, 20.0);
        let (delta, eff_n) = hierarchical_delta(10.0, 1.0, 0.0, 0.0, 0.0, 0.0, MATCHUP_PRIOR_K);
        assert!((delta - 1.0 / 30.0).abs() < 1e-12, "实际 {delta}");
        assert!(
            delta >= MIN_MATCHUP_ABS_DELTA,
            "10 场 6 胜的 3.3pp 残差在 1.5pp 门槛下应被导出，实际 {delta}"
        );
        assert!(
            passes_export_filter(delta, eff_n, MAP_MATCHUP_MIN_EFF_SAMPLES),
            "eff_n = 10 >= 8，应通过地图级门槛"
        );
    }

    #[test]
    fn thin_map_sample_is_blocked_by_sample_threshold() {
        // 4 场 3 胜：残差 1/24 ≈ 4.2pp 够显著，但 eff_n = 4 < 8 必须被样本门槛拦住
        let (delta, eff_n) = hierarchical_delta(4.0, 1.0, 0.0, 0.0, 0.0, 0.0, MATCHUP_PRIOR_K);
        assert!(delta.abs() >= MIN_MATCHUP_ABS_DELTA);
        assert!(
            !passes_export_filter(delta, eff_n, MAP_MATCHUP_MIN_EFF_SAMPLES),
            "地图级门槛为 eff_n >= 8，4 场不得导出"
        );
        // 同一份数据在模式级（eff_n >= 12）同样不够
        assert!(!passes_export_filter(delta, eff_n, MODE_MATCHUP_MIN_EFF_SAMPLES));
    }

    #[test]
    fn mode_threshold_is_stricter_than_map_threshold() {
        // eff_n = 10：地图级放行，模式级拦下——两层的稀疏程度不同
        let (delta, eff_n) = hierarchical_delta(10.0, 1.0, 0.0, 0.0, 0.0, 0.0, MATCHUP_PRIOR_K);
        assert_eq!(eff_n, 10.0);
        assert!(passes_export_filter(delta, eff_n, MAP_MATCHUP_MIN_EFF_SAMPLES));
        assert!(!passes_export_filter(delta, eff_n, MODE_MATCHUP_MIN_EFF_SAMPLES));
    }

    #[test]
    fn external_samples_are_discounted_and_capped() {
        // 同模式 10000 场 -> 折算 0.6 后封顶 40；其他模式 10000 场 -> 封顶 25
        let (delta, eff_n) =
            hierarchical_delta(0.0, 0.0, 10_000.0, 1_000.0, 10_000.0, 1_000.0, MATCHUP_PRIOR_K);
        assert!((eff_n - 65.0).abs() < 1e-9, "eff_n 应为 40 + 25，实际 {eff_n}");
        // e 同步按比例缩放：40/10000 * 1000 + 25/10000 * 1000 = 4 + 2.5
        assert!((delta - 6.5 / 85.0).abs() < 1e-12, "实际 {delta}");
    }

    #[test]
    fn large_local_sample_overrides_external_prior() {
        // 本图 200 场、+20 超额胜场（10% 残差）；同模式其他图 300 场却强烈反向
        let (delta, eff_n) = hierarchical_delta(200.0, 20.0, 300.0, -30.0, 0.0, 0.0, MATCHUP_PRIOR_K);
        assert!(eff_n > 200.0);
        assert!(
            delta > 0.05,
            "本图大样本必须能反转外部先验，实际 {delta}"
        );
        assert!(passes_export_filter(delta, eff_n, MAP_MATCHUP_MIN_EFF_SAMPLES));
    }

    #[test]
    fn thin_local_sample_defers_to_external_prior() {
        // 本图只有 10 场且略负，外部 300 场强正 -> 结果由先验主导
        let (delta, _) = hierarchical_delta(10.0, -1.0, 300.0, 30.0, 0.0, 0.0, MATCHUP_PRIOR_K);
        assert!(
            delta > MIN_MATCHUP_ABS_DELTA,
            "薄样本应被外部先验拉正，实际 {delta}"
        );
    }

    #[test]
    fn zero_external_samples_do_not_create_nan() {
        let (delta, eff_n) = hierarchical_delta(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, MATCHUP_PRIOR_K);
        assert!(delta.is_finite() && delta == 0.0);
        assert_eq!(eff_n, 0.0);
        assert!(!passes_export_filter(delta, eff_n, MAP_MATCHUP_MIN_EFF_SAMPLES));
    }

    // -----------------------------------------------------------------------
    // 全局兜底表
    // -----------------------------------------------------------------------

    #[test]
    fn global_matchups_keep_even_pairs_and_drop_thin_samples() {
        let mut excess: HashMap<PairKey, Stratum> = HashMap::new();
        // 均势对位：n 够、残差极小 -> 必须保留（前端据此判「五五开」）
        excess.insert((1, 2), Stratum { n: 400.0, e: 0.5 });
        // 薄样本：n < 10 -> 丢弃
        excess.insert((1, 3), Stratum { n: 9.0, e: 5.0 });
        // 强克制：n 够 -> 保留，delta = e / (n + K)
        excess.insert((2, 3), Stratum { n: 100.0, e: 20.0 });

        let out = build_global_matchups(&excess, MATCHUP_PRIOR_K);
        assert_eq!(out.len(), 2, "n < 10 的条目必须被丢弃，均势对位必须保留");

        // 按 (a, b) 升序，且恒有 a < b
        assert!(out.windows(2).all(|w| (w[0].a, w[0].b) < (w[1].a, w[1].b)));
        assert!(out.iter().all(|m| m.a < m.b));

        let even = out.iter().find(|m| (m.a, m.b) == (1, 2)).unwrap();
        assert_eq!(even.n, 400);
        assert!(
            even.delta.abs() < 0.5,
            "均势对位的残差应接近 0 但仍被导出（区别于无数据），实际 {}",
            even.delta
        );

        let strong = out.iter().find(|m| (m.a, m.b) == (2, 3)).unwrap();
        assert!((strong.delta - round2(20.0 / 120.0 * 100.0)).abs() < 1e-9);
        assert_eq!(strong.n, 100);
    }

    // -----------------------------------------------------------------------
    // 时间衰减
    // -----------------------------------------------------------------------

    #[test]
    fn decay_weight_halves_every_half_life() {
        assert!((decay_weight(0.0, 30.0) - 1.0).abs() < 1e-12);
        assert!((decay_weight(30.0, 30.0) - 0.5).abs() < 1e-12);
        assert!((decay_weight(60.0, 30.0) - 0.25).abs() < 1e-12);
        assert!((decay_weight(90.0, 30.0) - 0.125).abs() < 1e-12);
        // 未来时间（时钟偏差）不应放大权重
        assert!((decay_weight(-10.0, 30.0) - 1.0).abs() < 1e-12);
        // 非法半衰期退化为不衰减
        assert!((decay_weight(30.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((decay_weight(30.0, f64::NAN) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn win_rate_is_time_decay_weighted() {
        // 旧对局（30 天前，权重 0.5）A 赢；最新对局（权重 1.0）A 输
        // -> 加权场次 1.5、加权胜场 0.5 -> 加权胜率 1/3
        let battles = vec![
            row_at(
                "M",
                "brawlBall",
                "20261001T000000.000Z",
                [1, 10, 11],
                [2, 12, 13],
            ),
            row_at(
                "M",
                "brawlBall",
                "20261031T000000.000Z",
                [2, 12, 13],
                [1, 10, 11],
            ),
        ];

        let ids = vec![1, 2, 10, 11, 12, 13];
        let mut o = opts();
        o.decay_half_life_days = DEFAULT_DECAY_HALF_LIFE_DAYS;
        let out = analyze(&battles, &dict(&ids), &o);
        let a = out.maps[0].brawlers.iter().find(|b| b.id == 1).unwrap();

        // 原始计数不受影响
        assert_eq!(a.games, 2);
        assert_eq!(a.wins, 1);
        assert!(
            (a.win_rate - 1.0 / 3.0).abs() < 1e-3,
            "加权胜率应为 1/3，实际 {}",
            a.win_rate
        );
    }

    #[test]
    fn uniform_weights_reproduce_plain_win_rate() {
        // 每场对局都是独立样本（时间错开、衰减关闭）-> 权重全为 1.0，
        // 结果应与朴素胜率一致
        let mut battles = Vec::new();
        for i in 0..10 {
            let mut r = if i < 6 {
                row_at(
                    "M",
                    "brawlBall",
                    &shift_time(i as i64 * TEST_STEP_SECS),
                    [1, 10, 11],
                    [2, 12, 13],
                )
            } else {
                row_at(
                    "M",
                    "brawlBall",
                    &shift_time(i as i64 * TEST_STEP_SECS),
                    [2, 12, 13],
                    [1, 10, 11],
                )
            };
            r.battle_hash = format!("h{i}");
            battles.push(r);
        }
        let ids = vec![1, 2, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());
        let a = out.maps[0].brawlers.iter().find(|b| b.id == 1).unwrap();

        assert_eq!(a.games, 10);
        assert_eq!(a.wins, 6);
        assert!((a.win_rate - 0.6).abs() < 1e-9);
        // 平滑胜率 = (6 + 50*0.5) / (10 + 50) = 31/60
        assert!((a.smoothed_wr - 31.0 / 60.0).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // 段位过滤
    // -----------------------------------------------------------------------

    #[test]
    fn unknown_tier_is_kept_but_reported() {
        let mut battles = battles_with("M", "brawlBall", [1, 10, 11], [2, 12, 13], 4);
        for b in &mut battles {
            b.avg_tier = 0; // 官方 API 缺字段时的入库值
        }
        let ids = vec![1, 2, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());

        assert_eq!(out.total_battles, 4, "未知段位不应被静默清空");
        assert_eq!(out.tier_audit.unknown_tier_battles, 4);
        assert!(!out.tier_audit.filter_effective());
        assert_eq!(out.tier_audit.distribution.len(), 1);
        assert_eq!(out.tier_audit.distribution[0].tier, 0);
    }

    #[test]
    fn low_tier_battles_are_dropped() {
        let mut battles = battles_with("M", "brawlBall", [1, 10, 11], [2, 12, 13], 3);
        battles.extend(battles_with("M", "brawlBall", [2, 12, 13], [1, 10, 11], 3));
        for (i, b) in battles.iter_mut().enumerate() {
            b.avg_tier = if i < 3 { 12 } else { 30 };
            b.battle_hash = format!("h{i}");
        }
        let ids = vec![1, 2, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());

        assert_eq!(out.tier_audit.battles_total, 6);
        assert_eq!(out.total_battles, 3, "低于 min_tier 的对局应被剔除");
        assert!(out.tier_audit.filter_effective());
        assert_eq!(out.tier_audit.distribution.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 模式级克制
    // -----------------------------------------------------------------------

    #[test]
    fn mode_matchups_sum_excess_wins_across_maps() {
        // 两张同模式地图各 25 场阶梯（每层 25 胜 + 25 负 = 50 局）：
        // 对位样本 N_mode = 50，超额胜场 = 2 * 8.333，
        // delta = 16.67 / (50 + 20) ≈ 23.81pp（K 从 50 降到 20 后同步放大）
        let mut battles = counter_ladder("M1", "brawlBall", 25, "m1");
        battles.extend(counter_ladder("M2", "brawlBall", 25, "m2"));

        let out = analyze(&battles, &dict(&ALL), &opts());
        assert_eq!(out.modes.len(), 1, "同模式应聚合成一条");

        let mode = find_mode(&out, "brawlBall");
        assert_eq!(mode.total_games, 100);
        let a_vs_b = find_mode_brawler(mode, 1)
            .matchups
            .iter()
            .find(|m| m.opponent == 2)
            .expect("模式层应导出 A 对 B")
            .clone();
        assert_eq!(a_vs_b.samples, 50);
        assert!(
            (a_vs_b.delta - 23.81).abs() < 0.05,
            "模式层 delta 应为 23.81pp，实际 {}",
            a_vs_b.delta
        );
    }

    #[test]
    fn mode_delta_is_strictly_antisymmetric() {
        let battles = counter_ladder("M", "gemGrab", 25, "g");
        let out = analyze(&battles, &dict(&ALL), &opts());
        let mode = find_mode(&out, "gemGrab");

        let a_vs_b = find_mode_brawler(mode, 1)
            .matchups
            .iter()
            .find(|m| m.opponent == 2)
            .unwrap()
            .clone();
        let b_vs_a = find_mode_brawler(mode, 2)
            .matchups
            .iter()
            .find(|m| m.opponent == 1)
            .unwrap()
            .clone();
        assert_eq!(b_vs_a.samples, a_vs_b.samples);
        // 导出精度 2 位小数，反对称必须精确成立
        assert_eq!(b_vs_a.delta, -a_vs_b.delta);
    }

    #[test]
    fn strong_brawler_does_not_look_like_it_counters_everyone() {
        // 1 号是强势英雄：对所有人都有 60% 原始胜率；
        // 2/3 号是弱势英雄，对彼此 50%。Log5 残差应让 1 号的克制关系接近 0。
        let mut battles = Vec::new();
        let mut push = |i: usize, winners: [i64; 3], losers: [i64; 3], tag: &str| {
            let mut r = row_at(
                "M",
                "brawlBall",
                &shift_time(i as i64 * TEST_STEP_SECS),
                winners,
                losers,
            );
            r.battle_hash = format!("{tag}-{i}");
            battles.push(r);
        };
        for i in 0..60 {
            // 1 号赢 36 场 / 输 24 场（对 2 号与 3 号交替）
            let (win, lose) = if i % 2 == 0 {
                ([1, 10, 11], [2, 12, 13])
            } else {
                ([1, 10, 11], [3, 12, 13])
            };
            if i < 36 {
                push(i, win, lose, "w");
            } else {
                push(i, lose, win, "l");
            }
        }
        let ids = vec![1, 2, 3, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());
        let one = find_mode_brawler(find_mode(&out, "brawlBall"), 1);
        // 残差被平滑到门槛以下（条目不导出）或足够小，都说明没有被判成强力康特
        let delta = one
            .matchups
            .iter()
            .find(|m| m.opponent == 2)
            .map(|m| m.delta.abs())
            .unwrap_or(0.0);
        assert!(
            delta < 8.0,
            "基础强度高的英雄不应被判成强力康特，实际 delta={delta}"
        );
    }

    #[test]
    fn matchup_below_eff_threshold_is_dropped() {
        // 5 场 < 地图级 eff_n 门槛 8
        let battles = battles_with("M", "brawlBall", [1, 10, 11], [2, 12, 13], 5);
        let ids = vec![1, 2, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());
        let a = find_mode_brawler(find_mode(&out, "brawlBall"), 1);
        assert!(a.matchups.is_empty(), "5 场样本必须被剔除");
    }

    // -----------------------------------------------------------------------
    // 地图级克制 + 互斥分层
    // -----------------------------------------------------------------------

    #[test]
    fn map_matchup_needs_enough_eff_samples() {
        // 5 场：eff_n = 5 < 8，即使残差 9.1pp 也不导出
        let thin = counter_ladder("M", "brawlBall", 5, "t");
        let out = analyze(&thin, &dict(&ALL), &opts());
        assert!(
            matchup_on_map(find_map(&out, "M"), 1, 2).is_none(),
            "eff_n 不足的条目必须被剔除"
        );

        // 10 场：eff_n = 10 >= 8，残差 13.89pp，放行（旧口径 eff_n >= 15 会漏掉）
        let mid = counter_ladder("M", "brawlBall", 10, "m");
        let out = analyze(&mid, &dict(&ALL), &opts());
        let entry = matchup_on_map(find_map(&out, "M"), 1, 2).expect("10 场应导出");
        assert_eq!(entry.samples, 10);
        assert!(
            (entry.delta - 13.89).abs() < 0.05,
            "手算 delta 应为 13.89pp，实际 {}",
            entry.delta
        );

        // 25 场：eff_n = 25，delta = 18.52pp
        let solid = counter_ladder("M", "brawlBall", 25, "s");
        let out = analyze(&solid, &dict(&ALL), &opts());
        let entry = matchup_on_map(find_map(&out, "M"), 1, 2).expect("25 场应导出");
        assert_eq!(entry.samples, 25);
        assert!(
            (entry.delta - 18.52).abs() < 0.05,
            "手算 delta 应为 18.52pp，实际 {}",
            entry.delta
        );
    }

    #[test]
    fn map_delta_is_strictly_antisymmetric() {
        let battles = counter_ladder("M", "hotZone", 25, "h");
        let out = analyze(&battles, &dict(&ALL), &opts());
        let map = find_map(&out, "M");

        let ab = matchup_on_map(map, 1, 2).expect("A 对 B 应导出");
        let ba = matchup_on_map(map, 2, 1).expect("B 对 A 应导出");
        assert_eq!(ab.samples, ba.samples);
        assert_eq!(ab.delta, -ba.delta, "地图级同样必须严格反对称");
    }

    #[test]
    fn thin_map_sample_borrows_mode_prior_but_stays_below_local() {
        // M1 有 25 场阶梯，M2 只有 3 场：M2 的结论应被同模式先验抬起来，
        // 但仍低于本图样本充足的 M1（本图证据优先）。
        let mut battles = counter_ladder("M1", "brawlBall", 25, "m1");
        battles.extend(counter_ladder("M2", "brawlBall", 3, "m2"));

        let out = analyze(&battles, &dict(&ALL), &opts());
        let d1 = matchup_on_map(find_map(&out, "M1"), 1, 2).expect("M1 应导出");
        let d2 = matchup_on_map(find_map(&out, "M2"), 1, 2).expect("M2 应借先验导出");

        // 纯本图口径（无外部样本）时 M2 只有 3 场，残差约 6.2pp；
        // 借同模式先验后应更高，但仍低于本图样本充足的 M1
        assert!(
            d2.delta > 6.2,
            "薄样本应被同模式先验抬高，实际 {}",
            d2.delta
        );
        assert!(
            d2.delta < d1.delta,
            "本图样本不足时不应反超样本充足的图：M2={} M1={}",
            d2.delta,
            d1.delta
        );
    }

    #[test]
    fn ten_games_six_wins_clears_relaxed_threshold_end_to_end() {
        // 10 场 6 胜 4 负，双方只打过对方。平滑后的基础胜率是 0.5167 / 0.4833，
        // E_AB = 0.5333，e = 0.667，delta = 0.667 / 30 ≈ 2.22pp。
        // K 降到 20、门槛降到 1.5pp 后，这类中小样本**应当**导出——正是它让
        // 3×3 对位不再空白；旧口径（K=50 + eff_n>=15）会把它整条漏掉。
        let mut battles = Vec::new();
        for i in 0..10 {
            let t = shift_time(i as i64 * TEST_STEP_SECS);
            let mut r = if i < 6 {
                row_at("M", "bounty", &t, [1, 10, 11], [2, 12, 13])
            } else {
                row_at("M", "bounty", &t, [2, 12, 13], [1, 10, 11])
            };
            r.battle_hash = format!("n{i}");
            battles.push(r);
        }
        let out = analyze(&battles, &dict(&ALL), &opts());
        let entry = matchup_on_map(find_map(&out, "M"), 1, 2).expect("10 场 6 胜应导出");
        assert_eq!(entry.samples, 10);
        assert!(
            (entry.delta - 2.22).abs() < 0.05,
            "残差应为 2.22pp，实际 {}",
            entry.delta
        );
        // 反向条目必须严格取反
        let rev = matchup_on_map(find_map(&out, "M"), 2, 1).expect("反向条目同样导出");
        assert_eq!(rev.delta, -entry.delta);
    }

    #[test]
    fn strong_positive_residual_is_not_clamped() {
        // 后端严禁对正残差做抹零 / 截断：50 场阶梯的理论值 17.86pp 必须原样导出
        let battles = counter_ladder("M", "knockout", 50, "k");
        let out = analyze(&battles, &dict(&ALL), &opts());
        let entry = matchup_on_map(find_map(&out, "M"), 1, 2).expect("应导出");
        assert!(
            (entry.delta - 17.86).abs() < 0.05,
            "正残差必须原样保留，实际 {}",
            entry.delta
        );
    }

    #[test]
    fn per_map_base_rate_keeps_simpson_out_of_mode_layer() {
        // 辛普森陷阱：A 在两张图上都强于 B，但两图的整体强度水平不同。
        //
        // M1 是「高手图」：C 在该图极强（p_C ≈ 0.95），A 被 C 压着打；
        // M2 是「新手图」：A 只和 B 打。
        // 若先把两图的胜场混在一起再算 Log5，A 的混合基础胜率会被 M2 拉高，
        // 从而低估 A 对 B 的克制；按图分层去偏后，模式层应保留明显的正残差。
        let mut battles = counter_ladder("M1", "brawlBall", 25, "m1");
        battles.extend(battles_with(
            "M2",
            "brawlBall",
            [1, 10, 11],
            [2, 12, 13],
            25,
        ));
        for (i, b) in battles.iter_mut().enumerate() {
            if b.map_name == "M2" {
                b.battle_hash = format!("m2-{i}");
            }
        }

        let out = analyze(&battles, &dict(&ALL), &opts());
        let mode = find_mode(&out, "brawlBall");
        let entry = find_mode_brawler(mode, 1)
            .matchups
            .iter()
            .find(|m| m.opponent == 2)
            .expect("模式层应导出 A 对 B")
            .clone();
        assert!(
            entry.delta > 10.0,
            "按图去偏后 A 对 B 的克制不应被跨图混合稀释，实际 {}",
            entry.delta
        );
    }

    // -----------------------------------------------------------------------
    // 评分与顺位
    // -----------------------------------------------------------------------

    #[test]
    fn top_brawler_score_is_100_and_ordering_is_desc() {
        let battles = battles_with("M", "brawlBall", [1, 10, 11], [2, 12, 13], 30);
        let ids = vec![1, 2, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());
        let map = &out.maps[0];

        assert_eq!(map.brawlers[0].id, 1);
        assert!((map.brawlers[0].score - 100.0).abs() < 1e-9);
        assert!(map.brawlers.windows(2).all(|w| w[0].score >= w[1].score));
        assert!(map.brawlers.last().unwrap().score.abs() < 1e-9);
    }

    #[test]
    fn pick_orders_follow_quota_distribution() {
        // 10 个候选：2 个高波动、其余平稳 -> LAST 取 2、FIRST 取 2、MIDDLE 6
        let mut cands: Vec<DraftCandidate> = (0..10)
            .map(|i| {
                let (flex, spread) = match i {
                    0 => (0.55, 40.0),
                    1 => (0.52, 25.0),
                    2 => (0.70, 0.0),
                    3 => (0.65, 1.0),
                    _ => (0.45, 0.0),
                };
                dummy_candidate(i, flex, spread)
            })
            .collect();

        assign_pick_orders(&mut cands);

        let count = |p: PickOrder| cands.iter().filter(|c| c.stat.pick_order == p).count();
        assert_eq!(count(PickOrder::Last), 2, "quota_last = round(10 * 0.22) = 2");
        assert_eq!(count(PickOrder::First), 2, "quota_first = 2");
        assert_eq!(count(PickOrder::Middle), 6);

        // 波动最大的两个必须是 LAST
        assert_eq!(cands[0].stat.pick_order, PickOrder::Last);
        assert_eq!(cands[1].stat.pick_order, PickOrder::Last);
        // 剩余里 flex 最高的两个是 FIRST
        assert_eq!(cands[2].stat.pick_order, PickOrder::First);
        assert_eq!(cands[3].stat.pick_order, PickOrder::First);
    }

    #[test]
    fn flat_map_does_not_invent_counter_picks() {
        // 全部平稳（spread < 8pp）时不应硬凑 LAST，顺位只分 FIRST / MIDDLE
        let mut cands: Vec<DraftCandidate> = (0..10)
            .map(|i| dummy_candidate(i, 0.5 - i as f64 * 0.01, 2.0))
            .collect();
        assign_pick_orders(&mut cands);

        assert_eq!(
            cands
                .iter()
                .filter(|c| c.stat.pick_order == PickOrder::Last)
                .count(),
            0
        );
        assert_eq!(
            cands
                .iter()
                .filter(|c| c.stat.pick_order == PickOrder::First)
                .count(),
            2
        );
    }

    #[test]
    fn ineligible_heroes_never_get_first_or_last() {
        // 前 3 个样本不足（eligible=false），即便 flex / 波动最高也不能拿标签
        let mut cands: Vec<DraftCandidate> = (0..10)
            .map(|i| {
                let spread = if i < 3 { 40.0 } else { 20.0 - i as f64 };
                let mut c = dummy_candidate(i, 0.9 - i as f64 * 0.01, spread);
                c.eligible = i >= 3;
                c
            })
            .collect();
        assign_pick_orders(&mut cands);

        for c in cands.iter().take(3) {
            assert_eq!(
                c.stat.pick_order,
                PickOrder::Middle,
                "样本不足的英雄禁止标 FIRST / LAST"
            );
        }
        // 7 个合格英雄：LAST = round(7*0.22) = 2，FIRST = 2
        assert_eq!(
            cands
                .iter()
                .filter(|c| c.stat.pick_order == PickOrder::Last)
                .count(),
            2
        );
        assert_eq!(
            cands
                .iter()
                .filter(|c| c.stat.pick_order == PickOrder::First)
                .count(),
            2
        );
    }

    #[test]
    fn score_weights_win_rate_above_popularity() {
        // 热门卡：58% 出场、52.8% 胜率（Shooting Star 的 PIPER 量级）
        let popular = raw_score_of(1000, 0.58, 0.528, 0.59);
        // 冷门强卡：18% 出场、56.9% 胜率（MR. P 量级）
        let strong = raw_score_of(370, 0.18, 0.569, 0.59);
        assert!(
            strong > popular,
            "高胜率中等出场应压过高出场低胜率：strong={strong:.4} popular={popular:.4}"
        );
    }

    #[test]
    fn score_shrinks_tiny_samples_toward_neutral() {
        // 2 场全胜：即便平滑后胜率偏高，可靠性系数也会把胜率项拉回 50% 附近
        let tiny = raw_score_of(2, 0.01, 0.519, 0.59);
        let solid = raw_score_of(400, 0.18, 0.519, 0.59);
        assert!(
            tiny < solid,
            "同样胜率下，薄样本的综合分必须更低：tiny={tiny:.4} solid={solid:.4}"
        );
        // 薄样本的胜率项应该非常接近 0.5
        let rel = 2.0 / (2.0 + SCORE_RELIABILITY_K);
        let wr_core = 0.5 + (0.519 - 0.5) * rel;
        assert!((wr_core - 0.5).abs() < 0.002, "薄样本胜率项应贴近中性");
    }

    #[test]
    fn cold_hero_gets_low_score_and_no_first_label() {
        // 3 号只出场 2 场且全胜，其余英雄各 30+ 场
        let mut battles = battles_with("M", "bounty", [1, 10, 11], [2, 12, 13], 30);
        battles.extend(battles_with("M", "bounty", [3, 10, 11], [2, 12, 13], 2));

        let ids = vec![1, 2, 3, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());
        let map = &out.maps[0];

        let cold = map.brawlers.iter().find(|b| b.id == 3).unwrap();
        assert_eq!(cold.games, 2);
        assert!(
            cold.score <= SCORE_COLD_CAP,
            "极冷门英雄综合分必须封顶在 {SCORE_COLD_CAP}，实际 {}",
            cold.score
        );
        assert_eq!(
            cold.pick_order,
            PickOrder::Middle,
            "极冷门英雄禁止标 FIRST"
        );
        assert!(!is_pick_eligible(cold.games, cold.use_rate, 0));
    }

    #[test]
    fn eligible_range_keeps_top_scores_close() {
        // 榜首与次席的原始分差不应被出场率拉开成断崖
        let a = raw_score_of(400, 0.18, 0.569, 0.59);
        let b = raw_score_of(1000, 0.58, 0.528, 0.59);
        let gap = (a - b).abs();
        assert!(gap < 0.05, "顶部原始分差过大：{gap:.4}");
    }

    #[test]
    fn pick_order_uses_map_level_counters_not_mode_level() {
        // 同一模式（brawlBall）下的两张图：
        //   M1：1 号被 2 号完克（0/40）
        //   M2：1 号全胜（40/40），且没有任何劣势对位
        //
        // 若顺位用模式级克制，1 号会被 M1 的惨败拖累，在 M2 上拿不到先手；
        // 地图级判定下，M1 的克制不该污染 M2。
        let mut battles = Vec::new();
        for i in 0..40 {
            let mut r = row_at(
                "M1",
                "brawlBall",
                &shift_time(i as i64 * TEST_STEP_SECS),
                [2, 12, 13],
                [1, 10, 11],
            );
            r.battle_hash = format!("m1-{i}");
            battles.push(r);
        }
        for i in 0..40 {
            let mut r = row_at(
                "M2",
                "brawlBall",
                &shift_time(i as i64 * TEST_STEP_SECS),
                [1, 5, 6],
                [4, 14, 15],
            );
            r.battle_hash = format!("m2-{i}");
            battles.push(r);
        }

        let ids = vec![1, 2, 4, 5, 6, 10, 11, 12, 13, 14, 15];
        let out = analyze(&battles, &dict(&ids), &opts());

        let m2 = out.maps.iter().find(|m| m.map_name == "M2").unwrap();
        let one_on_m2 = m2.brawlers.iter().find(|b| b.id == 1).unwrap();
        assert_eq!(one_on_m2.games, 40);
        assert_eq!(
            one_on_m2.pick_order,
            PickOrder::First,
            "M2 上 1 号无劣势对位，应判先手（不应被 M1 的克制拖累）"
        );

        let m1 = out.maps.iter().find(|m| m.map_name == "M1").unwrap();
        let one_on_m1 = m1.brawlers.iter().find(|b| b.id == 1).unwrap();
        assert_ne!(
            one_on_m1.pick_order,
            PickOrder::First,
            "M1 上 1 号被完克，不应判先手"
        );
    }

    #[test]
    fn counter_pick_brawler_is_marked_last_on_map() {
        // 1 号稳吃 2 号，却被 3 号、4 号完克：对位画像呈明显两极（极差约 20pp），
        // 且有 3 条以上显著对位边，应判 LAST（后手康特位）。
        let mut battles = Vec::new();
        for i in 0..20 {
            let t = shift_time(i as i64 * TEST_STEP_SECS);

            let mut win = row_at("M", "brawlBall", &t, [1, 10, 11], [2, 12, 13]);
            win.battle_hash = format!("a{i}");
            battles.push(win);

            for (tag, counter) in [("b", 3), ("c", 4)] {
                let mut lose = row_at("M", "brawlBall", &t, [counter, 12, 13], [1, 10, 11]);
                lose.battle_hash = format!("{tag}{i}");
                battles.push(lose);
            }
        }
        let ids = vec![1, 2, 3, 4, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());
        let one = out.maps[0].brawlers.iter().find(|b| b.id == 1).unwrap();

        assert!(
            one.matchups.len() >= PICK_MIN_EDGES,
            "显著康特关系应被导出，实际 {} 条",
            one.matchups.len()
        );
        assert_eq!(one.pick_order, PickOrder::Last);
    }

    // -----------------------------------------------------------------------
    // 导出体积
    // -----------------------------------------------------------------------

    #[test]
    fn matchup_list_is_truncated_per_side() {
        // 20 个对手，全部 delta 为正 -> 只保留前 8 个优势，不浪费劣势名额
        let mut matchups: Vec<Matchup> = (0..20)
            .map(|i| Matchup {
                opponent: 100 + i,
                delta: 20.0 - i as f64,
                samples: 30,
                eff_samples: 30.0,
            })
            .collect();
        truncate_matchups(&mut matchups);
        assert_eq!(matchups.len(), MAP_MATCHUP_TOP_K);
        assert!(matchups.windows(2).all(|w| w[0].delta >= w[1].delta));
    }

    // -----------------------------------------------------------------------
    // BO3 系列聚簇降权
    // -----------------------------------------------------------------------

    /// 构造一局指定时间/阵容的对局（`battle_hash` 由调用方补）。
    fn battle_at(time: &str, winners: [i64; 3], losers: [i64; 3]) -> BattleRow {
        row_at("M", "brawlBall", time, winners, losers)
    }

    #[test]
    fn series_weights_merge_bo3_and_side_swap() {
        // 3 局同阵容、30 秒一局；第 4 局交换红蓝边且仍在 10 分钟内
        // -> 4 局同属一个系列，每局 1/4
        let mut battles = Vec::new();
        for (i, t) in [
            "20261020T100000.000Z",
            "20261020T100030.000Z",
            "20261020T100100.000Z",
        ]
        .iter()
        .enumerate()
        {
            let mut r = battle_at(t, [1, 10, 11], [2, 12, 13]);
            r.battle_hash = format!("w{i}");
            battles.push(r);
        }
        let mut swapped = battle_at("20261020T100130.000Z", [2, 12, 13], [1, 10, 11]);
        swapped.battle_hash = "swap".into();
        battles.push(swapped);

        // 换一张图：同阵容也不算同一系列
        let mut other_map = row_at("N", "brawlBall", "20261020T100200.000Z", [1, 10, 11], [2, 12, 13]);
        other_map.battle_hash = "other".into();
        battles.push(other_map);

        let refs: Vec<&BattleRow> = battles.iter().collect();
        let w = series_weights(&refs);
        for (i, v) in w.iter().take(4).enumerate() {
            assert!((v - 0.25).abs() < 1e-12, "第 {i} 局系列权重应为 1/4，实际 {v}");
        }
        assert!(
            (w[4] - 1.0).abs() < 1e-12,
            "不同地图必须独立成簇，实际 {}",
            w[4]
        );
    }

    #[test]
    fn series_window_is_ten_minutes_and_chains() {
        let mk = |tag: &str, t: &str| {
            let mut r = battle_at(t, [1, 10, 11], [2, 12, 13]);
            r.battle_hash = tag.into();
            r
        };

        // 恰好 600 秒：仍在同一系列
        let exact = [
            mk("a1", "20261020T100000.000Z"),
            mk("a2", "20261020T101000.000Z"),
        ];
        let refs: Vec<&BattleRow> = exact.iter().collect();
        assert!(series_weights(&refs).iter().all(|w| (w - 0.5).abs() < 1e-12));

        // 601 秒：拆成两个系列
        let split = [
            mk("b1", "20261020T100000.000Z"),
            mk("b2", "20261020T101001.000Z"),
        ];
        let refs: Vec<&BattleRow> = split.iter().collect();
        assert!(series_weights(&refs).iter().all(|w| (w - 1.0).abs() < 1e-12));

        // 链式合并：0s / 600s / 1200s，相邻两段都没超窗 -> 同一簇（3 局各 1/3）
        let chain = [
            mk("c1", "20261020T100000.000Z"),
            mk("c2", "20261020T101000.000Z"),
            mk("c3", "20261020T102000.000Z"),
        ];
        let refs: Vec<&BattleRow> = chain.iter().collect();
        let w = series_weights(&refs);
        assert!(w.iter().all(|v| (v - 1.0 / 3.0).abs() < 1e-12), "实际 {w:?}");
    }

    #[test]
    fn series_weight_is_multiplied_into_decay_weight() {
        let a = battle_at("20261020T100000.000Z", [1, 10, 11], [2, 12, 13]);
        let b = battle_at("20261020T100030.000Z", [1, 10, 11], [2, 12, 13]);
        let refs: Vec<&BattleRow> = vec![&a, &b];
        let w = weigh_battles(&refs, 30.0);

        // k = 2 -> 系列权重 0.5；最新局衰减 1.0，前一局衰减 0.5^(30/86400/30)
        assert!((w[1].1 - 0.5).abs() < 1e-12, "实际 {}", w[1].1);
        let decay = 0.5_f64.powf(30.0 / 86_400.0 / 30.0);
        assert!((w[0].1 - 0.5 * decay).abs() < 1e-12, "实际 {}", w[0].1);
    }

    #[test]
    fn bo3_series_counts_as_one_bp_group() {
        // 20 轮 BO3：每轮 3 局同阵容（30 秒一局），轮间间隔 1 小时
        let mut series = Vec::new();
        for i in 0..20i64 {
            for j in 0..3i64 {
                let mut r = battle_at(&shift_time(i * 3600 + j * 30), [1, 10, 11], [2, 12, 13]);
                r.battle_hash = format!("s{i}-{j}");
                series.push(r);
            }
        }
        // 对照组：20 场独立对局（间隔 > 10 分钟）
        let standalone = battles_with("M", "brawlBall", [1, 10, 11], [2, 12, 13], 20);

        let ids = vec![1, 2, 10, 11, 12, 13];
        let out_series = analyze(&series, &dict(&ids), &opts());
        let out_single = analyze(&standalone, &dict(&ids), &opts());

        let s = matchup_on_map(find_map(&out_series, "M"), 1, 2).expect("应导出");
        let c = matchup_on_map(find_map(&out_single, "M"), 1, 2).expect("应导出");

        assert_eq!(
            find_map(&out_series, "M").total_games,
            60,
            "地图总场次必须保留原始对局数"
        );
        assert_eq!(s.samples, 60, "原始对位样本数保留");
        assert!(
            (s.eff_samples - 20.0).abs() < 1e-9,
            "20 轮 BO3 的有效样本应为 20，实际 {}",
            s.eff_samples
        );
        // 一轮 3 局 BO3 与一场独立对局等价：有效样本与残差都必须一致
        assert!((s.eff_samples - c.eff_samples).abs() < 1e-9);
        assert!(
            (s.delta - c.delta).abs() < 1e-9,
            "BO3 降权后应与独立对局等价：{} vs {}",
            s.delta,
            c.delta
        );
    }

    #[test]
    fn two_game_series_halves_effective_samples() {
        // 10 轮 BO2（每轮 2 局）-> 有效样本 10，原始样本 20
        let mut battles = Vec::new();
        for i in 0..10i64 {
            for j in 0..2i64 {
                let mut r = battle_at(&shift_time(i * 3600 + j * 30), [1, 10, 11], [2, 12, 13]);
                r.battle_hash = format!("b{i}-{j}");
                battles.push(r);
            }
        }
        let out = analyze(&battles, &dict(&ALL), &opts());
        let entry = matchup_on_map(find_map(&out, "M"), 1, 2).expect("应导出");
        assert_eq!(entry.samples, 20);
        assert!(
            (entry.eff_samples - 10.0).abs() < 1e-9,
            "10 轮 BO2 的有效样本应为 10，实际 {}",
            entry.eff_samples
        );
    }

    #[test]
    fn series_cluster_report_counts_groups_not_battles() {
        // 3 局同阵容（30 秒一局）+ 1 场独立对局（换图）
        let mut battles = Vec::new();
        for (i, t) in [
            "20261020T100000.000Z",
            "20261020T100030.000Z",
            "20261020T100100.000Z",
        ]
        .iter()
        .enumerate()
        {
            let mut r = battle_at(t, [1, 10, 11], [2, 12, 13]);
            r.battle_hash = format!("s{i}");
            battles.push(r);
        }
        let mut solo = row_at("N", "brawlBall", "20261020T100200.000Z", [1, 10, 11], [2, 12, 13]);
        solo.battle_hash = "solo".into();
        battles.push(solo);

        let (raw, groups) = series_cluster_report(&battles, 0);
        assert_eq!(raw, 4, "原始对局数");
        assert_eq!(groups, 2, "1 个 BO3 系列 + 1 场独立对局 = 2 个有效 BP 组");
    }

    // -----------------------------------------------------------------------
    // 其他
    // -----------------------------------------------------------------------

    #[test]
    fn roles_are_resolved_to_ids_by_name() {
        let mut battles = vec![row("M", "brawlBall", [1, 10, 11], [2, 12, 13])];
        battles[0].battle_hash = "only".into();

        let mut d = HashMap::new();
        d.insert(1, "Shelly".to_string());
        d.insert(2, "Colt".to_string());

        let mut o = opts();
        o.roles.roles = vec![crate::models::output::RoleGroup {
            key: "TANK".into(),
            label: "坦克".into(),
            description: String::new(),
            brawlers: vec!["Shelly".into(), "NotExist".into()],
            brawler_ids: vec![],
        }];

        let out = analyze(&battles, &d, &o);
        assert_eq!(out.roles.roles[0].brawler_ids, vec![1]);
        assert_eq!(out.brawlers.len(), 6);
    }

    #[test]
    fn empty_input_produces_empty_output() {
        let out = analyze(&[], &HashMap::new(), &opts());
        assert!(out.maps.is_empty());
        assert!(out.modes.is_empty());
        assert!(out.global_matchups.is_empty());
        assert_eq!(out.total_battles, 0);
        assert_eq!(out.tier_audit.battles_total, 0);
    }
}
