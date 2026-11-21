//! 离线分析引擎：加权胜率、贝叶斯平滑、模式级克制矩阵与选人顺位。
//!
//! ## 两级聚合
//!
//! * **地图维度**（`maps[]`）：出场率、平滑胜率、综合评分、选人顺位。
//!   评分是「该图内的相对强度」，只能按图算。
//! * **模式维度**（`modes[]`）：1v1 克制矩阵。同一模式（`brawlBall` / `bounty` ...）
//!   下各图的克制关系高度一致，按模式聚合能把样本放大数倍——按 24 张单图切碎时，
//!   绝大多数对位样本会低于阈值而直接消失。
//!
//! ## 时间衰减
//!
//! 每个样本按 `w = 0.5^(Δt / 30)` 加权（`Δt` 为距**最新对局**的天数），
//! 使旧版本平衡性调整的数据自然降权，同时不必粗暴丢弃历史对局。
//!
//! ## 并行策略
//!
//! 先用 HashMap 按地图 / 模式分组（单线程 O(n)），再用 `rayon` 对**每张地图**、
//! **每个模式**并行聚合。地图与模式数量远大于核数，比逐对局并行更省同步开销。

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use rayon::prelude::*;

use crate::db::{parse_battle_time, BattleRow};
use crate::models::output::{
    AnalysisParams, BrawlerRef, MapBrawlerStat, MapMeta, Matchup, MetaOutput, ModeBrawlerStat,
    ModeMeta, PickOrder, RolesConfig, SCHEMA_VERSION,
};

/// 综合评分中平滑胜率的权重（其余给归一化出场率）。
const WR_WEIGHT: f64 = 0.75;
/// 综合评分中出场率的权重。
const USE_WEIGHT: f64 = 0.25;

/// 克制矩阵贝叶斯平滑的默认先验权重。
pub const DEFAULT_MATCHUP_PRIOR_WEIGHT: f64 = 6.0;
/// 时间衰减的默认半衰期（天）。
pub const DEFAULT_DECAY_HALF_LIFE_DAYS: f64 = 30.0;

/// 每个地图上被判为 `FIRST` 的比例（分位分配，保证分布稳定）。
const FIRST_RATIO: f64 = 0.22;
/// 每个地图上被判为 `LAST` 的比例上限。
const LAST_RATIO: f64 = 0.22;
/// 对位胜率极差低于该值，视为「没有明显康特关系」，不判 `LAST`。
const MIN_POLARITY: f64 = 0.08;
/// 存在明显劣势对位时的强度惩罚系数。
const FLEX_PENALTY: f64 = 0.30;
/// 低于该胜率开始扣分（视为「被康特」）。
const FLEX_PIVOT_WR: f64 = 0.48;
/// 单次扣分上限。
const FLEX_MAX_PENALTY: f64 = 0.30;
/// 计算「波动」至少需要的对位条数，单条对位不构成波动。
const MIN_MATCHUPS_FOR_PROFILE: usize = 2;

/// 分析参数。
#[derive(Debug, Clone)]
pub struct AnalyzeOptions {
    /// 胜率贝叶斯平滑的先验权重 m。
    pub bayes_prior_weight: f64,
    /// 克制矩阵贝叶斯平滑的先验权重 m（先验胜率固定 0.5）。
    pub matchup_prior_weight: f64,
    /// 克制矩阵最小交锋样本量（原始场次）。
    pub matchup_min_samples: u32,
    /// 时间衰减半衰期（天）。
    pub decay_half_life_days: f64,
    /// 段位阈值（仅回显到导出元数据）。
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
    let weighted = weigh_battles(battles, opts.decay_half_life_days);

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

    // 2) 模式维度并行聚合（克制矩阵）
    let mut modes: Vec<ModeMeta> = by_mode
        .par_iter()
        .map(|(mode, rows)| build_mode(mode, rows, brawler_dict, opts))
        .collect();
    modes.sort_by(|a, b| {
        b.total_games
            .cmp(&a.total_games)
            .then_with(|| a.mode.cmp(&b.mode))
    });

    // 3) 每个模式提取「最差对位 / 波动」，供地图顺位判定复用
    let profiles: HashMap<&str, HashMap<i64, MatchupProfile>> = modes
        .iter()
        .map(|m| (m.mode.as_str(), extract_profiles(m)))
        .collect();

    // 4) 地图维度并行聚合（评分 + 顺位）
    let mut maps: Vec<MapMeta> = by_map
        .par_iter()
        .map(|(map_name, rows)| build_map(map_name, rows, &profiles, brawler_dict, opts))
        .collect();
    maps.sort_by(|a, b| {
        b.total_games
            .cmp(&a.total_games)
            .then_with(|| a.map_name.cmp(&b.map_name))
    });

    MetaOutput {
        schema_version: SCHEMA_VERSION,
        updated_at: Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        season_start: opts.season_start.clone(),
        total_battles: battles.len() as u64,
        min_tier: opts.min_tier,
        params: AnalysisParams {
            bayes_prior_weight: opts.bayes_prior_weight,
            matchup_prior_weight: opts.matchup_prior_weight,
            matchup_min_samples: opts.matchup_min_samples,
            decay_half_life_days: opts.decay_half_life_days,
        },
        brawlers: build_brawler_refs(battles, brawler_dict),
        maps,
        modes,
        roles: resolve_roles(&opts.roles, brawler_dict),
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
fn weigh_battles(battles: &[BattleRow], half_life_days: f64) -> Vec<(&BattleRow, f64)> {
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
            (row, weight)
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

/// 模式内某个英雄的对位画像。
#[derive(Debug, Clone, Copy)]
struct MatchupProfile {
    /// 最差对位胜率（平滑后）。
    worst: f64,
    /// 对位胜率极差（平滑后）。
    spread: f64,
}

// ---------------------------------------------------------------------------
// 模式维度
// ---------------------------------------------------------------------------

fn build_mode(
    mode: &str,
    rows: &[(&BattleRow, f64)],
    dict: &HashMap<i64, String>,
    opts: &AnalyzeOptions,
) -> ModeMeta {
    let total_weight: f64 = rows.iter().map(|(_, w)| *w).sum();
    let mut acc: HashMap<i64, Accum> = HashMap::new();
    let mut pairs: HashMap<(i64, i64), Accum> = HashMap::new();

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
                let forward = pairs.entry((*a, *b)).or_default();
                forward.games += w;
                forward.wins += w;
                forward.raw_games += 1;
                forward.raw_wins += 1;

                let backward = pairs.entry((*b, *a)).or_default();
                backward.games += w;
                backward.raw_games += 1;
            }
        }
    }

    let min_samples = opts.matchup_min_samples as u64;
    let matchup_prior = opts.matchup_prior_weight.max(0.0);

    let mut brawlers: Vec<ModeBrawlerStat> = acc
        .iter()
        .map(|(id, a)| {
            let mut matchups: Vec<Matchup> = pairs
                .iter()
                .filter(|((from, _), p)| *from == *id && p.raw_games >= min_samples)
                .map(|((_, to), p)| Matchup {
                    opponent_id: *to,
                    games: p.raw_games,
                    win_rate: round4(ratio(p.wins, p.games)),
                    // 对位同样做贝叶斯平滑，避免 6~10 场的小样本出现 0% / 100%
                    smoothed_wr: round4(smoothed_rate(p.wins, p.games, matchup_prior)),
                })
                .collect();
            matchups.sort_by(|x, y| {
                y.games
                    .cmp(&x.games)
                    .then_with(|| x.opponent_id.cmp(&y.opponent_id))
            });

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
        total_games: rows.len() as u64,
        brawlers,
    }
}

/// 从模式结果中提取每个英雄的对位画像（最差对位胜率 + 极差）。
fn extract_profiles(mode: &ModeMeta) -> HashMap<i64, MatchupProfile> {
    let mut out = HashMap::new();
    for b in &mode.brawlers {
        if b.matchups.len() < MIN_MATCHUPS_FOR_PROFILE {
            continue;
        }
        let worst = b
            .matchups
            .iter()
            .map(|m| m.smoothed_wr)
            .fold(f64::INFINITY, f64::min);
        let best = b
            .matchups
            .iter()
            .map(|m| m.smoothed_wr)
            .fold(f64::NEG_INFINITY, f64::max);
        out.insert(
            b.id,
            MatchupProfile {
                worst,
                spread: best - worst,
            },
        );
    }
    out
}

// ---------------------------------------------------------------------------
// 地图维度
// ---------------------------------------------------------------------------

/// 参与顺位分配的地图英雄条目（`flex` 与 `spread` 只在分配阶段使用）。
struct DraftCandidate {
    stat: MapBrawlerStat,
    /// 综合强度：平滑胜率减去「被康特」惩罚。
    flex: f64,
    /// 同模式对位胜率极差。
    spread: f64,
}

fn build_map(
    map_name: &str,
    rows: &[(&BattleRow, f64)],
    profiles: &HashMap<&str, HashMap<i64, MatchupProfile>>,
    dict: &HashMap<i64, String>,
    opts: &AnalyzeOptions,
) -> MapMeta {
    let total_weight: f64 = rows.iter().map(|(_, w)| *w).sum();
    let mut acc: HashMap<i64, Accum> = HashMap::new();
    let mut mode_votes: HashMap<&str, u64> = HashMap::new();

    for (row, weight) in rows {
        *mode_votes.entry(row.mode.as_str()).or_insert(0) += 1;
        for b in &row.winners {
            acc.entry(*b).or_default().add_appearance(*weight, true);
        }
        for b in &row.losers {
            acc.entry(*b).or_default().add_appearance(*weight, false);
        }
    }

    let mode = mode_votes
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))
        .map(|(m, _)| m.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let empty_profiles: HashMap<i64, MatchupProfile> = HashMap::new();
    let mode_profiles = profiles.get(mode.as_str()).unwrap_or(&empty_profiles);

    let max_use = acc
        .values()
        .map(|a| ratio(a.games, total_weight))
        .fold(0.0_f64, f64::max);

    // 第一轮：算平滑胜率与未归一化的综合分
    let mut pre: Vec<(i64, Accum, f64, f64)> = acc
        .iter()
        .map(|(id, a)| {
            let use_rate = ratio(a.games, total_weight);
            let norm_use = if max_use > f64::EPSILON {
                use_rate / max_use
            } else {
                0.0
            };
            let smoothed = smoothed_rate(a.wins, a.games, opts.bayes_prior_weight);
            let raw_score = WR_WEIGHT * smoothed + USE_WEIGHT * norm_use;
            (*id, *a, smoothed, raw_score)
        })
        .collect();

    // 第二轮：Min-Max 缩放到 0~100（该图最高分英雄 = 100.0）
    let raw_min = pre.iter().map(|p| p.3).fold(f64::INFINITY, f64::min);
    let raw_max = pre
        .iter()
        .map(|p| p.3)
        .fold(f64::NEG_INFINITY, f64::max);
    let span = raw_max - raw_min;
    let degenerate = span <= 1e-12;

    let mut candidates: Vec<DraftCandidate> = pre
        .drain(..)
        .map(|(id, a, smoothed, raw_score)| {
            let score = if degenerate {
                // 只有一名英雄，或所有英雄指标完全一致：给中性分，避免虚高到 100
                50.0
            } else {
                (raw_score - raw_min) / span * 100.0
            };

            let (worst, spread) = match mode_profiles.get(&id) {
                Some(p) => (Some(p.worst), p.spread),
                None => (None, 0.0),
            };
            // 存在明显劣势对位时扣分：越差扣得越多，上限 0.09
            let penalty = worst
                .map(|w| (FLEX_PIVOT_WR - w).clamp(0.0, FLEX_MAX_PENALTY))
                .unwrap_or(0.0);
            let flex = smoothed - FLEX_PENALTY * penalty;

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
                },
                flex,
                spread,
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
/// 1. `LAST`：同模式对位极差最大的那一批（需达到 `MIN_POLARITY`，且至少 2 条对位）；
/// 2. `FIRST`：剩余英雄中 `flex` 最高的一批；
/// 3. 其余 `MIDDLE`。
///
/// 先判 `LAST` 再判 `FIRST`：波动极大的英雄本质上不适合先手。
fn assign_pick_orders(candidates: &mut [DraftCandidate]) {
    let n = candidates.len();
    if n == 0 {
        return;
    }
    let quota_first = (((n as f64) * FIRST_RATIO).round() as usize).clamp(1, n);
    let quota_last = (((n as f64) * LAST_RATIO).round() as usize).clamp(1, n);

    // 1) LAST：波动大且确有康特关系
    let mut by_spread: Vec<usize> = (0..n).collect();
    by_spread.sort_by(|&a, &b| {
        candidates[b]
            .spread
            .partial_cmp(&candidates[a].spread)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| candidates[a].stat.id.cmp(&candidates[b].stat.id))
    });
    let mut last_taken = 0usize;
    for &i in &by_spread {
        if last_taken >= quota_last {
            break;
        }
        if candidates[i].spread >= MIN_POLARITY {
            candidates[i].stat.pick_order = PickOrder::Last;
            last_taken += 1;
        }
    }

    // 2) FIRST：剩余英雄里强度最高的
    let mut rest: Vec<usize> = (0..n)
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
// 工具函数
// ---------------------------------------------------------------------------

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
fn build_brawler_refs(battles: &[BattleRow], dict: &HashMap<i64, String>) -> Vec<BrawlerRef> {
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
            matchup_prior_weight: DEFAULT_MATCHUP_PRIOR_WEIGHT,
            matchup_min_samples: 6,
            decay_half_life_days: DEFAULT_DECAY_HALF_LIFE_DAYS,
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

    fn dummy_candidate(id: i64, flex: f64, spread: f64) -> DraftCandidate {
        DraftCandidate {
            stat: MapBrawlerStat {
                id,
                name: format!("B{id}"),
                games: 10,
                wins: 5,
                win_rate: 0.5,
                smoothed_wr: 0.5,
                use_rate: 0.1,
                score: 50.0,
                pick_order: PickOrder::Middle,
            },
            flex,
            spread,
        }
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
    // 模式级克制矩阵
    // -----------------------------------------------------------------------

    #[test]
    fn matchups_aggregate_across_maps_of_same_mode() {
        // A(1) 与 B(2) 在两张同模式地图上各交锋 3 场，合计 6 场
        // 按单图切碎时各自只有 3 场（低于阈值 6），按模式聚合后应达标
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
            .find(|m| m.opponent_id == 2)
            .expect("跨图聚合后应达到 6 场阈值");
        assert_eq!(a_vs_b.games, 6);
        assert!((a_vs_b.win_rate - 1.0).abs() < 1e-9);
        // 6 场全胜也必须被平滑，不能出现 100%
        assert!(
            a_vs_b.smoothed_wr < 0.95,
            "小样本应被平滑，实际 {}",
            a_vs_b.smoothed_wr
        );
        assert!((a_vs_b.smoothed_wr - (6.0 + 3.0) / 12.0).abs() < 1e-4);
    }

    #[test]
    fn matchup_matrix_is_symmetric() {
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
        let a_vs_b = a.matchups.iter().find(|m| m.opponent_id == 2).unwrap();
        assert_eq!(a_vs_b.games, 20);
        assert!((a_vs_b.win_rate - 0.6).abs() < 1e-9);

        let b = find_mode_brawler(mode, 2);
        let b_vs_a = b.matchups.iter().find(|m| m.opponent_id == 1).unwrap();
        assert_eq!(b_vs_a.games, 20);
        assert!((b_vs_a.win_rate - 0.4).abs() < 1e-9);
        // 双方样本量一致，平滑后应严格互补
        assert!((a_vs_b.smoothed_wr + b_vs_a.smoothed_wr - 1.0).abs() < 1e-4);
    }

    #[test]
    fn matchup_below_threshold_is_dropped() {
        // 5 场 < 阈值 6
        let mut battles = Vec::new();
        for i in 0..5 {
            let mut r = row("M", "brawlBall", [1, 10, 11], [2, 12, 13]);
            r.battle_hash = format!("h{i}");
            battles.push(r);
        }
        let ids = vec![1, 2, 10, 11, 12, 13];
        let out = analyze(&battles, &dict(&ids), &opts());
        let a = find_mode_brawler(find_mode(&out, "brawlBall"), 1);
        assert!(a.matchups.is_empty(), "5 场样本必须被剔除");
    }

    // -----------------------------------------------------------------------
    // 评分与顺位
    // -----------------------------------------------------------------------

    #[test]
    fn top_brawler_score_is_100_and_ordering_is_desc() {
        let mut battles = Vec::new();
        for i in 0..30 {
            let mut r = row("M", "brawlBall", [1, 10, 11], [2, 12, 13]);
            r.battle_hash = format!("h{i}");
            battles.push(r);
        }
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
                    0 => (0.55, 0.40),
                    1 => (0.52, 0.25),
                    2 => (0.70, 0.0),
                    3 => (0.65, 0.01),
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
        // 全部平稳（spread < 0.08）时不应硬凑 LAST，顺位只分 FIRST / MIDDLE
        let mut cands: Vec<DraftCandidate> = (0..10)
            .map(|i| dummy_candidate(i, 0.5 - i as f64 * 0.01, 0.02))
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
    fn counter_pick_brawler_is_marked_last_on_map() {
        // B1 对 B2 大优、对 B3 大劣 -> 波动极大 -> LAST
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
    }
}
