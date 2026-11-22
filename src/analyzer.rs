//! 离线分析引擎：加权胜率、贝叶斯平滑、Log5 残差克制矩阵与选人顺位。
//!
//! ## 两级聚合
//!
//! * **地图维度**（`maps[]`）：出场率、平滑胜率、综合评分、选人顺位，以及
//!   **地图级克制列表**。同一英雄在不同地形下（草图 vs 开阔图）克制关系差异很大，
//!   因此地图层是克制结论的首选来源。
//! * **模式维度**（`modes[]`）：同模式下各图的克制关系高度一致，按模式聚合能把
//!   样本放大数倍，作为地图样本不足时的**先验**。
//!
//! ## 克制算法：Log5 预期胜率残差 + 贝叶斯平滑
//!
//! 直接用「A 打 B 的原始胜率」当克制分是错的：强势英雄对谁都能赢，会被误判成
//! 「克制所有人」。正确做法是先扣掉双方的基础强度：
//!
//! ```text
//! p_A    = clamp((wins_A + m/2) / (games_A + m), 0.05, 0.95)   // 平滑基础胜率
//! E_AB   = p_A(1-p_B) / (p_A(1-p_B) + (1-p_A)p_B)              // Log5 预期对位胜率
//! delta  = (w_AB - n_AB * E_AB) / (n_AB + C) * 100             // 净胜率差值（pp）
//! ```
//!
//! * `C` 是平滑常数（伪场次，建议 15~20），小样本的残差被压向 0；
//! * 因为 `E_BA = 1 - E_AB` 且 `w_BA = n - w_AB`，所以 `delta_BA = -delta_AB` 严格成立。
//!
//! ### 层级兜底：地图优先，模式先验
//!
//! 单图对位样本往往很薄。地图层用「留一图」的模式残差做先验，并按样本量连续收缩：
//!
//! ```text
//! prior_pp  = (w_mode_other - n_mode_other * E_mode) / (n_mode_other + C) * 100
//! prior_w   = C * max(0, 1 - n_map / TRUST)          // TRUST 场后完全信任单图
//! delta_map = (w_map - n_map*E_map + prior_w*prior_pp/100) / (n_map + C) * 100
//! ```
//!
//! `n_map >= TRUST` 时 `prior_w = 0`，退化为纯单图估计；`n_map = 0` 时完全等于模式先验。
//! 收缩权重随样本量连续衰减，避免在阈值附近出现跳变。
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
//! **克制层例外**：`delta` 使用原始计数，保证与公式一致、可手算复核。
//!
//! ## 并行策略
//!
//! 先用 HashMap 按地图 / 模式分组（单线程 O(n)），再用 `rayon` 对**每张地图**、
//! **每个模式**并行聚合。

use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use rayon::prelude::*;

use crate::db::{parse_battle_time, BattleRow};
use crate::models::output::{
    tier_label, AnalysisParams, BrawlerRef, MapBrawlerStat, MapMeta, Matchup, MetaOutput,
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

/// 克制残差贝叶斯平滑的默认常数 C（伪场次）。
pub const DEFAULT_MATCHUP_SMOOTHING_C: f64 = 15.0;
/// 时间衰减的默认半衰期（天）。
pub const DEFAULT_DECAY_HALF_LIFE_DAYS: f64 = 30.0;
/// 地图级克制导出的最小原始对位场次。
pub const DEFAULT_MAP_MATCHUP_MIN_SAMPLES: u32 = 5;
/// 地图级对位达到该场次后完全信任单图数据。
pub const DEFAULT_MAP_MATCHUP_TRUST_SAMPLES: u32 = 15;

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

/// 分析参数。
#[derive(Debug, Clone)]
pub struct AnalyzeOptions {
    /// 胜率贝叶斯平滑的先验权重 m。
    pub bayes_prior_weight: f64,
    /// 克制残差贝叶斯平滑常数 C（伪场次）。
    pub matchup_smoothing_c: f64,
    /// 模式级克制导出的最小原始对位场次。
    pub matchup_min_samples: u32,
    /// 段位阈值：仅保留 `avg_tier >= min_tier` 的对局（`avg_tier <= 0` 视为未知，保留）。
    pub min_tier: i64,
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

    let weighted = weigh_battles(&kept, DEFAULT_DECAY_HALF_LIFE_DAYS);

    // 1) 按地图 / 模式分组（一次遍历，两边共用同一份权重）
    let mut by_map: HashMap<&str, Vec<(&BattleRow, f64)>> = HashMap::new();
    let mut by_mode: HashMap<&str, Vec<(&BattleRow, f64)>> = HashMap::new();
    for (row, weight) in &weighted {
        by_map
            .entry(row.map_name.as_str())
            .or_default()
            .push((*row, *weight));
        by_mode
            .entry(row.mode.as_str())
            .or_default()
            .push((*row, *weight));
    }

    // 2) 模式维度并行聚合（克制矩阵 + 地图层的先验）
    let mut built: Vec<(ModeMeta, ModeContext)> = by_mode
        .par_iter()
        .map(|(mode, rows)| build_mode(mode, rows, brawler_dict, opts))
        .collect();
    built.sort_by(|a, b| {
        b.0.total_games
            .cmp(&a.0.total_games)
            .then_with(|| a.0.mode.cmp(&b.0.mode))
    });

    let contexts: HashMap<&str, &ModeContext> =
        built.iter().map(|(m, c)| (m.mode.as_str(), c)).collect();

    // 3) 地图维度并行聚合（评分 + 顺位 + 地图级克制）
    //
    // 顺位判定用的是**本图**的对位画像，不再复用模式级结果：同一英雄在草图与
    // 开阔图上的克制差异很大，拿模式级最差对位去评一张开阔图，会被无关地形的
    // 近战克制污染（例如狙击图上的投掷手被按「撞脸坦克」扣分）。
    let mut maps: Vec<MapMeta> = by_map
        .par_iter()
        .map(|(map_name, rows)| build_map(map_name, rows, &contexts, brawler_dict, opts))
        .collect();
    maps.sort_by(|a, b| {
        b.total_games
            .cmp(&a.total_games)
            .then_with(|| a.map_name.cmp(&b.map_name))
    });

    let modes: Vec<ModeMeta> = built.into_iter().map(|(m, _)| m).collect();

    MetaOutput {
        schema_version: SCHEMA_VERSION,
        updated_at: Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        season_start: opts.season_start.clone(),
        total_battles: kept.len() as u64,
        min_tier: opts.min_tier,
        tier_audit,
        params: AnalysisParams {
            bayes_prior_weight: opts.bayes_prior_weight,
            matchup_smoothing_c: opts.matchup_smoothing_c,
            matchup_min_samples: opts.matchup_min_samples,
            map_matchup_min_samples: DEFAULT_MAP_MATCHUP_MIN_SAMPLES,
            map_matchup_trust_samples: DEFAULT_MAP_MATCHUP_TRUST_SAMPLES,
            decay_half_life_days: DEFAULT_DECAY_HALF_LIFE_DAYS,
        },
        brawlers: build_brawler_refs(&kept, brawler_dict),
        maps,
        modes,
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

/// 给每条对局计算权重。基准是**数据集中最新的对局**，而不是 `Utc::now()`，
/// 这样离线重算历史数据时结果可复现。
fn weigh_battles<'a>(
    battles: &[&'a BattleRow],
    half_life_days: f64,
) -> Vec<(&'a BattleRow, f64)> {
    let times: Vec<Option<DateTime<Utc>>> = battles
        .iter()
        .map(|b| parse_battle_time(&b.battle_time))
        .collect();
    let latest = times.iter().flatten().copied().max();

    battles
        .iter()
        .zip(times)
        .map(|(row, time)| {
            let weight = match (latest, time) {
                (Some(latest), Some(t)) => {
                    let delta_days = (latest - t).num_seconds() as f64 / 86_400.0;
                    decay_weight(delta_days, half_life_days)
                }
                // 时间缺失时不做衰减，按满权重处理（入库前已校验过时间，属于兜底）
                _ => 1.0,
            };
            (*row, weight)
        })
        .collect()
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
}

/// 两两对位的原始计数。
#[derive(Debug, Default, Clone, Copy)]
struct PairStat {
    /// 真实交锋场次。
    n: u64,
    /// 前者对后者的胜场。
    w: u64,
}

impl PairStat {
    fn add_win(&mut self) {
        self.n += 1;
        self.w += 1;
    }

    fn add_loss(&mut self) {
        self.n += 1;
    }
}

/// 模式内某个英雄的对位画像（单位均为百分点）。
#[derive(Debug, Clone, Copy)]
struct MatchupProfile {
    /// 最差对位 delta（负数 = 被克制）。
    worst_pp: f64,
    /// 对位 delta 极差。
    spread_pp: f64,
}

/// 模式级的克制上下文：给地图层做「留一图」先验。
///
/// 只保留先验需要的两样东西——平滑基础胜率与两两原始计数；
/// 模式级的 delta 矩阵在 `build_mode` 内部算完直接用于导出，不再驻留。
struct ModeContext {
    /// 平滑基础胜率（已截断到 `[0.05, 0.95]`）。
    base: HashMap<i64, f64>,
    /// 两两对位原始计数（双向各一条，保证 `w_ab + w_ba = n`）。
    pairs: HashMap<(i64, i64), PairStat>,
}

// ---------------------------------------------------------------------------
// Log5 数学
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

/// 贝叶斯平滑后的净胜率差值（pp）。
///
/// `delta = (w - n*E) / (n + C) * 100`：分子是真实胜场与预期胜场之差，
/// 分母加 `C` 个伪场次，把小样本残差压向 0。
pub fn residual_delta(wins: u64, games: u64, expected: f64, c: f64) -> f64 {
    if games == 0 {
        return 0.0;
    }
    let n = games as f64;
    ((wins as f64 - n * expected) / (n + c)) * 100.0
}

/// 地图级 delta：单图残差 + 模式先验的连续收缩。
fn map_delta(
    games: u64,
    wins: u64,
    expected_map: f64,
    prior_pp: f64,
    c: f64,
    trust: f64,
) -> f64 {
    let n = games as f64;
    let resid = wins as f64 - n * expected_map;
    // 样本越多越信任单图；达到 trust 后先验权重归零，与纯单图公式连续衔接。
    let prior_w = c * (1.0 - n / trust).max(0.0);
    ((resid + prior_w * prior_pp / 100.0) / (n + c)) * 100.0
}

// ---------------------------------------------------------------------------
// 模式维度
// ---------------------------------------------------------------------------

fn build_mode(
    mode: &str,
    rows: &[(&BattleRow, f64)],
    dict: &HashMap<i64, String>,
    opts: &AnalyzeOptions,
) -> (ModeMeta, ModeContext) {
    let total_weight: f64 = rows.iter().map(|(_, w)| *w).sum();
    let mut acc: HashMap<i64, Accum> = HashMap::new();
    let mut pairs: HashMap<(i64, i64), PairStat> = HashMap::new();

    for (row, weight) in rows {
        let w = *weight;
        for b in &row.winners {
            acc.entry(*b).or_default().add_appearance(w, true);
        }
        for b in &row.losers {
            acc.entry(*b).or_default().add_appearance(w, false);
        }
        for a in &row.winners {
            for b in &row.losers {
                if a == b {
                    continue; // 镜像对局不计入克制关系
                }
                pairs.entry((*a, *b)).or_default().add_win();
                pairs.entry((*b, *a)).or_default().add_loss();
            }
        }
    }

    let c = opts.matchup_smoothing_c.max(0.0);
    let base: HashMap<i64, f64> = acc
        .iter()
        .map(|(id, a)| {
            let p = smoothed_rate(
                a.raw_wins as f64,
                a.raw_games as f64,
                opts.bayes_prior_weight,
            );
            (*id, clamp_base(p))
        })
        .collect();

    // 只按 (min, max) 方向算一次再镜像，保证 delta_BA == -delta_AB 精确成立
    let mut delta: HashMap<(i64, i64), f64> = HashMap::with_capacity(pairs.len());
    for ((a, b), p) in &pairs {
        if a >= b {
            continue;
        }
        let e = log5(base.get(a).copied().unwrap_or(0.5), base.get(b).copied().unwrap_or(0.5));
        let d = residual_delta(p.w, p.n, e, c);
        delta.insert((*a, *b), d);
        delta.insert((*b, *a), -d);
    }

    let min_samples = opts.matchup_min_samples as u64;
    let mut brawlers: Vec<ModeBrawlerStat> = acc
        .iter()
        .map(|(id, a)| {
            let mut matchups: Vec<Matchup> = pairs
                .iter()
                .filter(|((from, _), p)| *from == *id && p.n >= min_samples)
                .map(|((_, to), p)| Matchup {
                    opponent: *to,
                    delta: round2(delta.get(&(*id, *to)).copied().unwrap_or(0.0)),
                    samples: p.n,
                })
                .collect();
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

    (
        ModeMeta {
            mode: mode.to_string(),
            total_games: rows.len() as u64,
            brawlers,
        },
        ModeContext { base, pairs },
    )
}

/// 从一组对位统计中提取每个英雄的对位画像（最差 delta + 极差）。
///
/// 只统计样本达标的对位边；不足 [`MIN_MATCHUPS_FOR_PROFILE`] 条边的英雄不产生画像
/// （没有画像就不会被判 `LAST`，也不会被「被康特」惩罚扣分）。
fn extract_profiles(
    pairs: &HashMap<(i64, i64), PairStat>,
    delta: &HashMap<(i64, i64), f64>,
    min_samples: u64,
) -> HashMap<i64, MatchupProfile> {
    let mut agg: HashMap<i64, (f64, f64, usize)> = HashMap::new();
    for ((from, to), p) in pairs {
        if p.n < min_samples {
            continue;
        }
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
    map_name: &str,
    rows: &[(&BattleRow, f64)],
    contexts: &HashMap<&str, &ModeContext>,
    dict: &HashMap<i64, String>,
    opts: &AnalyzeOptions,
) -> MapMeta {
    let total_weight: f64 = rows.iter().map(|(_, w)| *w).sum();
    let mut acc: HashMap<i64, Accum> = HashMap::new();
    let mut pairs: HashMap<(i64, i64), PairStat> = HashMap::new();
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
                    continue;
                }
                pairs.entry((*a, *b)).or_default().add_win();
                pairs.entry((*b, *a)).or_default().add_loss();
            }
        }
    }

    let mode = mode_votes
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))
        .map(|(m, _)| m.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let c = opts.matchup_smoothing_c.max(0.0);
    let trust = DEFAULT_MAP_MATCHUP_TRUST_SAMPLES as f64;

    let base: HashMap<i64, f64> = acc
        .iter()
        .map(|(id, a)| {
            let p = smoothed_rate(
                a.raw_wins as f64,
                a.raw_games as f64,
                opts.bayes_prior_weight,
            );
            (*id, clamp_base(p))
        })
        .collect();

    let ctx = contexts.get(mode.as_str()).copied();

    // 地图级 delta：单图残差为主，样本不足时向「留一图」的模式残差收缩
    let mut delta: HashMap<(i64, i64), f64> = HashMap::with_capacity(pairs.len());
    for ((a, b), p) in &pairs {
        if a >= b {
            continue;
        }
        let p_a = base.get(a).copied().unwrap_or(0.5);
        let p_b = base.get(b).copied().unwrap_or(0.5);
        let e_map = log5(p_a, p_b);

        let prior_pp = match ctx {
            Some(cx) => {
                let mode_pair = cx.pairs.get(&(*a, *b)).copied().unwrap_or_default();
                // 留一图：扣掉本图的贡献，避免先验里重复计入同一批对局
                let n_other = mode_pair.n.saturating_sub(p.n);
                let w_other = mode_pair.w.saturating_sub(p.w);
                if n_other == 0 {
                    0.0
                } else {
                    let e_mode = log5(
                        cx.base.get(a).copied().unwrap_or(p_a),
                        cx.base.get(b).copied().unwrap_or(p_b),
                    );
                    residual_delta(w_other, n_other, e_mode, c)
                }
            }
            None => 0.0,
        };

        let d = map_delta(p.n, p.w, e_map, prior_pp, c, trust);
        delta.insert((*a, *b), d);
        delta.insert((*b, *a), -d);
    }

    // 本图对位画像：顺位判定只看这张图上的克制关系
    let min_export = DEFAULT_MAP_MATCHUP_MIN_SAMPLES as u64;
    let map_profiles = extract_profiles(&pairs, &delta, min_export);

    let max_use = acc
        .values()
        .map(|a| ratio(a.games, total_weight))
        .fold(0.0_f64, f64::max);

    // 第一轮：算平滑胜率、出场率与未归一化的综合分
    let mut pre: Vec<(i64, Accum, f64, f64)> = acc
        .iter()
        .map(|(id, a)| {
            let use_rate = ratio(a.games, total_weight);
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
        .filter(|(_, a, _, _)| is_score_eligible(a.raw_games, ratio(a.games, total_weight)))
        .map(|p| p.3)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    let span = raw_max - raw_min;
    let degenerate = !span.is_finite() || span <= 1e-12;

    let mut candidates: Vec<DraftCandidate> = pre
        .drain(..)
        .map(|(id, a, smoothed, raw_score)| {
            let use_rate = ratio(a.games, total_weight);
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
                .map(|w| {
                    ((FLEX_PIVOT_PP - w) / FLEX_MAX_PENALTY_PP).clamp(0.0, 1.0)
                })
                .unwrap_or(0.0);
            let flex = smoothed - FLEX_PENALTY * penalty;

            let mut matchups: Vec<Matchup> = pairs
                .iter()
                .filter(|((from, _), p)| *from == id && p.n >= min_export)
                .map(|((_, to), p)| Matchup {
                    opponent: *to,
                    delta: round2(delta.get(&(id, *to)).copied().unwrap_or(0.0)),
                    samples: p.n,
                })
                .collect();
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
                    use_rate: round4(ratio(a.games, total_weight)),
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
        map_name: map_name.to_string(),
        mode,
        total_games: rows.len() as u64,
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
            matchup_smoothing_c: DEFAULT_MATCHUP_SMOOTHING_C,
            matchup_min_samples: 6,
            min_tier: 20,
            season_start: "2026-07-20".into(),
            roles: RolesConfig {
                version: 1,
                roles: vec![],
            },
        }
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

    fn find_mode_brawler(mode: &ModeMeta, id: i64) -> &ModeBrawlerStat {
        mode.brawlers
            .iter()
            .find(|b| b.id == id)
            .expect("英雄应存在")
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
    fn battles_with(
        map: &str,
        mode: &str,
        winners: [i64; 3],
        losers: [i64; 3],
        n: usize,
    ) -> Vec<BattleRow> {
        (0..n)
            .map(|i| {
                let mut r = row(map, mode, winners, losers);
                r.battle_hash = format!("{map}-{i}");
                r
            })
            .collect()
    }

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

    #[test]
    fn residual_delta_shrinks_small_samples() {
        // 5 场全胜、E=0.5 -> 分子 2.5；C=15 -> 2.5/20*100 = 12.5pp
        assert!((residual_delta(5, 5, 0.5, 15.0) - 12.5).abs() < 1e-9);
        // 同样全胜但 50 场 -> 25/65*100 ≈ 38.46pp
        assert!((residual_delta(50, 50, 0.5, 15.0) - 25.0 / 65.0 * 100.0).abs() < 1e-9);
        // 符合预期时残差为 0
        assert!(residual_delta(10, 20, 0.5, 15.0).abs() < 1e-12);
    }

    #[test]
    fn residual_delta_is_antisymmetric() {
        let n = 40u64;
        let w = 26u64;
        let e = log5(0.58, 0.47);
        let ab = residual_delta(w, n, e, 15.0);
        let ba = residual_delta(n - w, n, 1.0 - e, 15.0);
        assert!((ab + ba).abs() < 1e-12, "delta_BA 必须是 -delta_AB");
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
        let out = analyze(&battles, &dict(&ids), &opts());
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
        // 所有对局时间相同 -> 权重全为 1.0，结果应与朴素胜率一致
        let mut battles = Vec::new();
        for i in 0..10 {
            let mut r = if i < 6 {
                row("M", "brawlBall", [1, 10, 11], [2, 12, 13])
            } else {
                row("M", "brawlBall", [2, 12, 13], [1, 10, 11])
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
    // 模式级克制矩阵
    // -----------------------------------------------------------------------

    #[test]
    fn matchups_aggregate_across_maps_of_same_mode() {
        // A(1) 与 B(2) 在两张同模式地图上各交锋 3 场，合计 6 场
        // 按单图切碎时各自只有 3 场（低于导出阈值 6），按模式聚合后应达标
        let mut battles = Vec::new();
        for (map, count) in [("M1", 3), ("M2", 3)] {
            for i in 0..count {
                let mut r = row(map, "brawlBall", [1, 10, 11], [2, 12, 13]);
                r.battle_hash = format!("{map}-{i}");
                battles.push(r);
            }
        }

        let ids = vec![1, 2, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());

        assert_eq!(out.maps.len(), 2, "地图维度仍按图切分");
        assert_eq!(out.modes.len(), 1, "同模式应聚合成一条");

        let mode = find_mode(&out, "brawlBall");
        assert_eq!(mode.total_games, 6);
        let a = find_mode_brawler(mode, 1);
        let a_vs_b = a
            .matchups
            .iter()
            .find(|m| m.opponent == 2)
            .expect("跨图聚合后应达到 6 场阈值");
        assert_eq!(a_vs_b.samples, 6);
        // 6 场全胜，但基础胜率也高 -> 残差必须远小于「100% - 预期」
        assert!(a_vs_b.delta > 0.0, "全胜应表现为正 delta");
        assert!(a_vs_b.delta < 50.0, "小样本残差应被 C 平滑");
    }

    #[test]
    fn matchup_delta_is_strictly_antisymmetric() {
        let mut battles = Vec::new();
        for i in 0..20 {
            let mut r = if i < 12 {
                row("M", "gemGrab", [1, 10, 11], [2, 12, 13])
            } else {
                row("M", "gemGrab", [2, 12, 13], [1, 10, 11])
            };
            r.battle_hash = format!("h{i}");
            battles.push(r);
        }

        let ids = vec![1, 2, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());
        let mode = find_mode(&out, "gemGrab");

        let a = find_mode_brawler(mode, 1);
        let a_vs_b = a.matchups.iter().find(|m| m.opponent == 2).unwrap();
        assert_eq!(a_vs_b.samples, 20);
        assert!(a_vs_b.delta > 0.0);

        let b = find_mode_brawler(mode, 2);
        let b_vs_a = b.matchups.iter().find(|m| m.opponent == 1).unwrap();
        assert_eq!(b_vs_a.samples, 20);
        // 导出精度 2 位小数，反对称必须精确成立
        assert_eq!(b_vs_a.delta, -a_vs_b.delta);
    }

    #[test]
    fn strong_brawler_does_not_look_like_it_counters_everyone() {
        // 1 号是强势英雄：对所有人都有 60% 原始胜率；
        // 2/3 号是弱势英雄，对彼此 50%。Log5 残差应让 1 号的克制关系接近 0。
        let mut battles = Vec::new();
        let mut push = |i: usize, winners: [i64; 3], losers: [i64; 3], tag: &str| {
            let mut r = row("M", "brawlBall", winners, losers);
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
        let mode = find_mode(&out, "brawlBall");
        let one = find_mode_brawler(mode, 1);
        let vs_two = one.matchups.iter().find(|m| m.opponent == 2).unwrap();
        assert!(
            vs_two.delta.abs() < 8.0,
            "基础强度高的英雄不应被判成强力康特，实际 delta={}",
            vs_two.delta
        );
    }

    #[test]
    fn matchup_below_threshold_is_dropped() {
        // 5 场 < 阈值 6
        let battles = battles_with("M", "brawlBall", [1, 10, 11], [2, 12, 13], 5);
        let ids = vec![1, 2, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());
        let a = find_mode_brawler(find_mode(&out, "brawlBall"), 1);
        assert!(a.matchups.is_empty(), "5 场样本必须被剔除");
    }

    // -----------------------------------------------------------------------
    // 地图级克制 + 层级收缩
    // -----------------------------------------------------------------------

    #[test]
    fn map_matchups_exist_at_lower_threshold_than_mode() {
        // 单图 8 场：满足地图阈值 5，可导出
        let battles = battles_with("M", "brawlBall", [1, 10, 11], [2, 12, 13], 8);
        let ids = vec![1, 2, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());
        let map = &out.maps[0];
        let a = map.brawlers.iter().find(|b| b.id == 1).unwrap();
        let vs_b = a.matchups.iter().find(|m| m.opponent == 2).unwrap();
        assert_eq!(vs_b.samples, 8);
        assert!(vs_b.delta > 0.0);
    }

    #[test]
    fn map_delta_shrinks_toward_mode_prior_until_trust() {
        let c = DEFAULT_MATCHUP_SMOOTHING_C;
        let trust = DEFAULT_MAP_MATCHUP_TRUST_SAMPLES as f64;
        let prior = 20.0; // 模式先验：+20pp

        // 3 场全胜：单图残差 1.5；先验权重 15*(1-3/15)=12
        let thin = map_delta(3, 3, 0.5, prior, c, trust);
        let pure_thin = map_delta(3, 3, 0.5, 0.0, c, trust);
        assert!(
            thin > pure_thin,
            "薄样本应被模式先验拉高：{thin} 应大于纯单图 {pure_thin}"
        );

        // 达到 trust 场后先验权重归零，等价于纯单图公式
        let full = map_delta(15, 12, 0.5, prior, c, trust);
        let pure_full = map_delta(15, 12, 0.5, 0.0, c, trust);
        assert!((full - pure_full).abs() < 1e-12, "满 trust 后应完全信任单图");

        // 0 场时退化为纯先验
        assert!((map_delta(0, 0, 0.5, prior, c, trust) - prior).abs() < 1e-12);
    }

    #[test]
    fn thin_map_matchup_is_dropped_from_export() {
        // 3 场 < 地图导出阈值 5，不应出现在地图级列表里
        let mut battles = battles_with("M1", "brawlBall", [1, 10, 11], [2, 12, 13], 30);
        battles.extend(battles_with("M2", "brawlBall", [1, 10, 11], [2, 12, 13], 3));

        let ids = vec![1, 2, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());

        let m2 = out.maps.iter().find(|m| m.map_name == "M2").unwrap();
        let a = m2.brawlers.iter().find(|b| b.id == 1).unwrap();
        assert!(
            a.matchups.iter().all(|m| m.opponent != 2),
            "3 场对位应被导出阈值过滤"
        );
    }

    #[test]
    fn map_matchup_is_antisymmetric() {
        let battles = battles_with("M", "hotZone", [1, 10, 11], [2, 12, 13], 20);
        let ids = vec![1, 2, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());
        let map = &out.maps[0];
        let a = map.brawlers.iter().find(|b| b.id == 1).unwrap();
        let b = map.brawlers.iter().find(|b| b.id == 2).unwrap();
        let ab = a.matchups.iter().find(|m| m.opponent == 2).unwrap();
        let ba = b.matchups.iter().find(|m| m.opponent == 1).unwrap();
        assert_eq!(ab.samples, ba.samples);
        assert_eq!(ab.delta, -ba.delta, "地图级同样必须严格反对称");
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
            cands.iter().filter(|c| c.stat.pick_order == PickOrder::Last).count(),
            2
        );
        assert_eq!(
            cands.iter().filter(|c| c.stat.pick_order == PickOrder::First).count(),
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
            let mut r = row("M1", "brawlBall", [2, 12, 13], [1, 10, 11]);
            r.battle_hash = format!("m1-{i}");
            battles.push(r);
        }
        for i in 0..40 {
            let mut r = row("M2", "brawlBall", [1, 5, 6], [4, 14, 15]);
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
        let mut battles = Vec::new();
        for i in 0..20 {
            let mut win = row("M", "brawlBall", [1, 10, 11], [2, 12, 13]);
            win.battle_hash = format!("a{i}");
            battles.push(win);

            let mut lose = row("M", "brawlBall", [3, 12, 13], [1, 10, 11]);
            lose.battle_hash = format!("b{i}");
            battles.push(lose);
        }
        let ids = vec![1, 2, 3, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());
        let one = out.maps[0].brawlers.iter().find(|b| b.id == 1).unwrap();

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
            })
            .collect();
        truncate_matchups(&mut matchups);
        assert_eq!(matchups.len(), MAP_MATCHUP_TOP_K);
        assert!(matchups.windows(2).all(|w| w[0].delta >= w[1].delta));
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
        assert_eq!(out.total_battles, 0);
        assert_eq!(out.tier_audit.battles_total, 0);
    }
}
