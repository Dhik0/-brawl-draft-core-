//! 逼真的模拟对局生成器。
//!
//! 用途：在没有 API Key 的情况下跑通「采集 -> 清洗 -> 分析 -> 导出」全链路，
//! 同时给 `analyzer` 的数学逻辑提供有真实信号（地图专精、强度差异）的样本。
//!
//! 实现上刻意**复用真实的入库路径** `Db::ingest_battle`，
//! 这样 mock 数据与线上数据走完全相同的清洗、去重与胜方判定逻辑。

use anyhow::{Context, Result};
use chrono::Duration as ChronoDuration;
use tracing::info;

use super::{format_battle_time, Db, IngestFilter, IngestOutcome};
use crate::models::api::{Battle, BattleItem, BattlePlayer, Brawler, Event};

/// 模拟英雄名册（ID 从 16000000 顺序递增，与官方 ID 段一致）。
const ROSTER: &[&str] = &[
    "Shelly", "Colt", "Bull", "Brock", "El Primo", "Barley", "Poco", "Rosa", "Jessie", "Nita",
    "Dynamike", "Tick", "8-Bit", "Rico", "Darryl", "Penny", "Carl", "Jacky", "Bo", "Emz", "Stu",
    "Piper", "Pam", "Frank", "Bibi", "Bea", "Nani", "Edgar", "Griff", "Grom", "Gale", "Colette",
    "Belle", "Ash", "Lola", "Sam", "Mandy", "Maisie", "Hank", "Pearl", "Angelo", "Draco", "Lily",
    "Buzz", "Mortis", "Leon", "Crow", "Sprout", "Tara", "Gene", "Sandy", "Lou", "Otis", "Charlie",
    "Buster", "Ruffs", "Byron", "Berry", "Mico", "Kit", "Larry & Laurie", "Juju", "Clancy",
];

/// 地图定义：(地图名, 玩法, 强势英雄名)。
const MAPS: &[(&str, &str, &[&str])] = &[
    (
        "Center Stage",
        "brawlBall",
        &["Bull", "El Primo", "Colt", "Brock", "Buzz", "Rosa", "Frank", "Gale"],
    ),
    (
        "Pinball Dreams",
        "brawlBall",
        &["Bull", "Darryl", "Shelly", "Gale", "Mortis", "Frank", "Buzz", "Jacky"],
    ),
    (
        "Double Swoosh",
        "gemGrab",
        &["Piper", "Belle", "Gene", "Tara", "Emz", "Penny", "Bo", "Pam"],
    ),
    (
        "Hard Rock Mine",
        "gemGrab",
        &["Tara", "Gene", "Sandy", "Pam", "Colt", "Lou", "Emz", "Bo"],
    ),
    (
        "Hot Potato",
        "hotZone",
        &["Emz", "Lou", "Sandy", "Poco", "Pam", "Gale", "Barley", "Buster"],
    ),
    (
        "Bridge Too Far",
        "bounty",
        &["Piper", "Bea", "Belle", "Mandy", "Nani", "Angelo", "Brock", "Crow"],
    ),
];

/// 模拟玩家池规模。
const PLAYER_POOL: usize = 600;
/// 强势英雄的抽样权重，普通英雄为 1。
const META_WEIGHT: u64 = 6;

/// 生成结果统计。
#[derive(Debug, Clone, Copy, Default)]
pub struct MockSummary {
    pub requested: usize,
    pub inserted: usize,
    pub duplicates: usize,
    pub filtered: usize,
    pub players: usize,
}

/// 极简 xorshift64* 伪随机数发生器（避免引入白名单外的 `rand` 依赖）。
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next_u64() % n
        }
    }

    fn chance(&mut self, p: f64) -> bool {
        let r = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        r < p
    }
}

const TAG_ALPHABET: &[u8] = b"0289PYLQGRJCUV";

fn random_tag(rng: &mut Rng) -> String {
    let mut s = String::with_capacity(10);
    s.push('#');
    for _ in 0..9 {
        s.push(TAG_ALPHABET[rng.below(TAG_ALPHABET.len() as u64) as usize] as char);
    }
    s
}

/// 生成 `count` 场模拟排位对局并写入数据库。
///
/// `season_start` 用于把对局时间铺在赛季开始之后的 45 天内，
/// 保证不会因为「早于赛季」被清洗掉。
pub fn generate(
    db: &mut Db,
    count: usize,
    seed: u64,
    filter: &IngestFilter,
) -> Result<MockSummary> {
    let mut rng = Rng::new(seed);
    let mut summary = MockSummary {
        requested: count,
        ..Default::default()
    };

    // ---- 玩家池 ----
    let mut pool: Vec<(String, i64)> = Vec::with_capacity(PLAYER_POOL);
    for _ in 0..PLAYER_POOL {
        // 段位 20~35：刚好覆盖默认 MIN_TIER=20 的边界
        let rank = 20 + rng.below(16) as i64;
        pool.push((random_tag(&mut rng), rank));
    }
    for (tag, rank) in &pool {
        db.set_player_tier(tag, *rank)?;
    }
    summary.players = pool.len();

    // ---- 每张地图的加权抽样表 ----
    let weighted: Vec<Vec<(usize, u64)>> = MAPS
        .iter()
        .map(|(_, _, strong)| {
            let strong_idx: Vec<usize> = strong
                .iter()
                .filter_map(|name| ROSTER.iter().position(|r| r == name))
                .collect();
            ROSTER
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let w = if strong_idx.contains(&i) { META_WEIGHT } else { 1 };
                    (i, w)
                })
                .collect()
        })
        .collect();

    let base_time = filter.season_start + ChronoDuration::days(1);

    for _ in 0..count {
        let map_idx = rng.below(MAPS.len() as u64) as usize;
        let (map_name, mode, _) = MAPS[map_idx];
        let table = &weighted[map_idx];

        // 6 个互不重复的英雄
        let mut picks: Vec<usize> = Vec::with_capacity(6);
        while picks.len() < 6 {
            let pick = weighted_pick(&mut rng, table);
            if !picks.contains(&pick) {
                picks.push(pick);
            }
        }

        // 6 名互不重复的玩家
        let mut player_idx: Vec<usize> = Vec::with_capacity(6);
        while player_idx.len() < 6 {
            let idx = rng.below(pool.len() as u64) as usize;
            if !player_idx.contains(&idx) {
                player_idx.push(idx);
            }
        }

        // 队伍强度 = 英雄权重之和，决定胜率倾向
        let weight_of = |roster_idx: usize| -> f64 {
            table
                .iter()
                .find(|(i, _)| *i == roster_idx)
                .map(|(_, w)| *w as f64)
                .unwrap_or(1.0)
        };
        let strength_a: f64 = picks[..3].iter().map(|i| weight_of(*i)).sum();
        let strength_b: f64 = picks[3..].iter().map(|i| weight_of(*i)).sum();
        let diff = (strength_a - strength_b) / (strength_a + strength_b);
        let p_a_wins = 0.5 + 0.28 * diff;
        let a_wins = rng.chance(p_a_wins);

        // 对局时间：赛季开始后 1~45 天内
        let offset_secs = rng.below(45 * 24 * 3600) as i64;
        let battle_time = format_battle_time(&(base_time + ChronoDuration::seconds(offset_secs)));

        let teams: Vec<Vec<BattlePlayer>> = (0..2)
            .map(|team| {
                (0..3)
                    .map(|slot| {
                        let p = &pool[player_idx[team * 3 + slot]];
                        let roster_idx = picks[team * 3 + slot];
                        BattlePlayer {
                            tag: p.0.clone(),
                            name: Some(format!("Player{}", &p.0[1..5])),
                            rank: Some(p.1),
                            brawler: Brawler {
                                id: brawler_id(roster_idx),
                                name: Some(ROSTER[roster_idx].to_string()),
                                power: Some(11),
                                trophies: Some(600 + p.1 * 40),
                                rank: Some(p.1),
                            },
                        }
                    })
                    .collect()
            })
            .collect();

        let item = BattleItem {
            battle_time,
            event: Event {
                id: Some(15000000 + map_idx as i64),
                mode: Some("soloRanked".into()),
                map: Some(map_name.to_string()),
            },
            battle: Battle {
                mode: Some(mode.to_string()),
                battle_type: Some("soloRanked".into()),
                // battlelog 的视角是第一名玩家（A 队 1 号位）
                result: Some(if a_wins { "victory" } else { "defeat" }.into()),
                teams: Some(teams),
                ..Default::default()
            },
            battle_type: None,
        };

        match db
            .ingest_battle(&item, &pool[player_idx[0]].0, filter)
            .context("写入模拟对局失败")?
        {
            IngestOutcome::Inserted(_) => summary.inserted += 1,
            IngestOutcome::Duplicate => summary.duplicates += 1,
            IngestOutcome::Filtered(_) => summary.filtered += 1,
        }
    }

    info!(
        requested = summary.requested,
        inserted = summary.inserted,
        duplicates = summary.duplicates,
        filtered = summary.filtered,
        players = summary.players,
        "模拟数据生成完成"
    );
    Ok(summary)
}

/// 按权重抽样一个英雄下标。
fn weighted_pick(rng: &mut Rng, table: &[(usize, u64)]) -> usize {
    let total: u64 = table.iter().map(|(_, w)| *w).sum();
    let mut roll = rng.below(total.max(1));
    for (idx, w) in table {
        if roll < *w {
            return *idx;
        }
        roll -= *w;
    }
    table.last().map(|(i, _)| *i).unwrap_or(0)
}

/// 名册下标 -> 英雄 ID。
pub fn brawler_id(roster_idx: usize) -> i64 {
    16_000_000 + roster_idx as i64
}

/// 名册（供测试与文档使用）。
pub fn roster() -> &'static [&'static str] {
    ROSTER
}

/// 地图定义（供测试与文档使用）。
pub fn maps() -> &'static [(&'static str, &'static str, &'static [&'static str])] {
    MAPS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;
    use chrono::Utc;

    fn filter() -> IngestFilter {
        IngestFilter {
            min_tier: 20,
            season_start: chrono::DateTime::parse_from_rfc3339("2026-09-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        }
    }

    #[test]
    fn roster_and_maps_are_consistent() {
        for (map, mode, strong) in maps() {
            assert!(!map.is_empty() && !mode.is_empty());
            for name in *strong {
                assert!(roster().contains(name), "{map} 引用了不存在的英雄 {name}");
            }
        }
        // ID 唯一
        let mut ids: Vec<i64> = (0..roster().len()).map(brawler_id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), roster().len());
    }

    #[test]
    fn generates_requested_battles_without_duplicates() {
        let mut db = Db::open_in_memory().unwrap();
        let summary = generate(&mut db, 300, 42, &filter()).unwrap();

        assert_eq!(summary.requested, 300);
        assert_eq!(summary.inserted, 300, "模拟数据不应触发清洗过滤");
        assert_eq!(summary.duplicates, 0);
        assert_eq!(db.stats().unwrap().battles, 300);
        assert_eq!(db.stats().unwrap().players, PLAYER_POOL as i64);
    }

    #[test]
    fn generation_is_deterministic_for_same_seed() {
        let mut a = Db::open_in_memory().unwrap();
        let mut b = Db::open_in_memory().unwrap();
        generate(&mut a, 120, 7, &filter()).unwrap();
        generate(&mut b, 120, 7, &filter()).unwrap();

        let ha: Vec<String> = {
            let mut v: Vec<String> = a
                .load_battles()
                .unwrap()
                .into_iter()
                .map(|r| r.battle_hash)
                .collect();
            v.sort();
            v
        };
        let hb: Vec<String> = {
            let mut v: Vec<String> = b
                .load_battles()
                .unwrap()
                .into_iter()
                .map(|r| r.battle_hash)
                .collect();
            v.sort();
            v
        };
        assert_eq!(ha, hb, "同一 seed 必须产出同一批对局");
    }

    #[test]
    fn battles_land_after_season_start_and_have_valid_teams() {
        let mut db = Db::open_in_memory().unwrap();
        generate(&mut db, 80, 2026, &filter()).unwrap();

        let season = filter().season_start;
        for row in db.load_battles().unwrap() {
            let t = crate::db::parse_battle_time(&row.battle_time).expect("时间应可解析");
            assert!(t >= season, "对局时间 {} 早于赛季开始", row.battle_time);
            assert_eq!(row.winners.len(), 3);
            assert_eq!(row.losers.len(), 3);
            assert!(!row.map_name.is_empty());
        }
    }
}
