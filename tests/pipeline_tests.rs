//! 端到端管线测试：去重、清洗、mock 生成与（阶段 3 起）统计导出。
//!
//! 这些用例刻意走公共 API，模拟真实调用顺序：多份 battlelog -> 入库去重 -> 分析导出。

use brawl_draft_core::analyzer::{analyze, AnalyzeOptions};
use brawl_draft_core::db::{compute_battle_hash, parse_battle_time, Db, IngestFilter, IngestOutcome};
use brawl_draft_core::models::api::{Battle, BattleItem, BattlePlayer, Brawler, Event};
use brawl_draft_core::models::output::{MetaOutput, PickOrder, RolesConfig};
use chrono::{DateTime, Utc};
use std::path::Path;

fn filter() -> IngestFilter {
    IngestFilter {
        min_tier: 20,
        season_start: DateTime::parse_from_rfc3339("2026-09-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc),
    }
}

fn mk_player(tag: &str, brawler_id: i64) -> BattlePlayer {
    BattlePlayer {
        tag: tag.to_string(),
        name: Some(format!("name{tag}")),
        rank: Some(28),
        brawler: Brawler {
            id: brawler_id,
            name: Some(format!("Brawler{brawler_id}")),
            power: Some(11),
            trophies: Some(800),
            rank: Some(28),
        },
    }
}

/// 构造一局 3v3 排位：`observer_won` 表示 battlelog 所属玩家的胜负。
fn mk_battle(observer_won: bool) -> BattleItem {
    BattleItem {
        battle_time: "20261020T143015.000Z".to_string(),
        event: Event {
            id: Some(15000001),
            mode: Some("soloRanked".into()),
            map: Some("Center Stage".into()),
        },
        battle: Battle {
            mode: Some("brawlBall".into()),
            battle_type: Some("soloRanked".into()),
            result: Some(if observer_won { "victory" } else { "defeat" }.into()),
            teams: Some(vec![
                vec![
                    mk_player("#AAA111", 16_000_000),
                    mk_player("#BBB222", 16_000_001),
                    mk_player("#CCC333", 16_000_002),
                ],
                vec![
                    mk_player("#DDD444", 16_000_003),
                    mk_player("#EEE555", 16_000_004),
                    mk_player("#FFF666", 16_000_005),
                ],
            ]),
            ..Default::default()
        },
        battle_type: None,
    }
}

#[test]
fn six_battlelogs_collapse_into_one_battle() {
    let mut db = Db::open_in_memory().unwrap();
    let f = filter();

    // A 队三人看到 victory，B 队三人看到 defeat；无论谁先入库，都只应留一行
    let team_a = ["#AAA111", "#BBB222", "#CCC333"];
    let team_b = ["#DDD444", "#EEE555", "#FFF666"];

    let mut inserted = 0;
    for tag in team_a {
        let outcome = db.ingest_battle(&mk_battle(true), tag, &f).unwrap();
        if matches!(outcome, IngestOutcome::Inserted(_)) {
            inserted += 1;
        }
    }
    for tag in team_b {
        let outcome = db.ingest_battle(&mk_battle(false), tag, &f).unwrap();
        assert_eq!(outcome, IngestOutcome::Duplicate);
    }

    assert_eq!(inserted, 1);
    assert_eq!(db.stats().unwrap().battles, 1);

    let row = db.load_battles().unwrap().remove(0);
    // 胜负判定与观察者无关：A 队始终是胜方
    assert_eq!(row.winners, vec![16_000_000, 16_000_001, 16_000_002]);
    assert_eq!(row.losers, vec![16_000_003, 16_000_004, 16_000_005]);
}

#[test]
fn hash_matches_regardless_of_tag_order() {
    let a = ["#AAA111", "#BBB222", "#CCC333", "#DDD444", "#EEE555", "#FFF666"];
    let mut b: Vec<String> = a.iter().map(|s| s.to_string()).collect();
    b.sort();
    b.reverse();

    let h1 = compute_battle_hash("20261020T143015.000Z", &a.map(str::to_string));
    let h2 = compute_battle_hash("20261020T143015.000Z", &b);
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 64);
}

#[test]
fn battle_time_round_trips() {
    let dt = parse_battle_time("20261020T143015.000Z").expect("应能解析");
    assert_eq!(dt.to_rfc3339(), "2026-10-20T14:30:15+00:00");
    assert!(parse_battle_time("2026-10-20 14:30").is_none());
    assert!(parse_battle_time("").is_none());
}

#[test]
fn mock_data_feeds_the_database() {
    let mut db = Db::open_in_memory().unwrap();
    let summary = brawl_draft_core::db::mock::generate(&mut db, 500, 2026, &filter()).unwrap();

    assert_eq!(summary.inserted, 500);
    let stats = db.stats().unwrap();
    assert_eq!(stats.battles, 500);
    assert!(stats.maps >= 4, "应覆盖多张地图，实际 {}", stats.maps);
    assert!(stats.players > 0);

    let dict = db.brawler_dict().unwrap();
    assert!(dict.len() >= 30, "英雄字典应包含足够样本");

    // 每局都是 3v3，且胜败双方不重叠
    for row in db.load_battles().unwrap() {
        assert_eq!(row.winners.len(), 3);
        assert_eq!(row.losers.len(), 3);
        for w in &row.winners {
            assert!(!row.losers.contains(w), "同一英雄不应同时出现在两方");
        }
    }
}

#[test]
fn mock_then_analyze_produces_valid_meta() {
    let mut db = Db::open_in_memory().unwrap();
    let f = filter();
    let summary = brawl_draft_core::db::mock::generate(&mut db, 2000, 99, &f).unwrap();
    assert_eq!(summary.inserted, 2000);

    let battles = db.load_battles().unwrap();
    let dict = db.brawler_dict().unwrap();
    let roles = RolesConfig::load(Path::new("config/brawler_roles.json")).unwrap();

    let opts = AnalyzeOptions {
        bayes_prior_weight: 50.0,
        matchup_prior_weight: 6.0,
        matchup_min_samples: 6,
        decay_half_life_days: 30.0,
        min_tier: 20,
        season_start: "2026-07-20".into(),
        roles,
    };
    let out = analyze(&battles, &dict, &opts);

    assert_eq!(out.total_battles, 2000);
    assert!(out.maps.len() >= 4, "应覆盖多张地图");
    assert!(!out.modes.is_empty(), "应产出模式维度结果");
    assert!(!out.brawlers.is_empty());
    assert!(
        out.roles.roles.iter().any(|r| !r.brawler_ids.is_empty()),
        "定位标签应能解析出英雄 ID"
    );

    // ---- 地图维度 ----
    for map in &out.maps {
        assert!(map.total_games > 0);
        assert!(!map.brawlers.is_empty());
        // 每局 6 个英雄出场位
        let slots: u64 = map.brawlers.iter().map(|b| b.games).sum();
        assert_eq!(slots, map.total_games * 6, "{} 出场位统计不闭合", map.map_name);

        // 每图最高分归一化为 100
        let top = &map.brawlers[0];
        assert!((top.score - 100.0).abs() < 1e-9, "{} 最高分应为 100", map.map_name);
        assert!(map.brawlers.windows(2).all(|w| w[0].score >= w[1].score));

        for b in &map.brawlers {
            assert!((0.0..=1.0).contains(&b.use_rate));
            assert!((0.0..=1.0).contains(&b.win_rate));
            assert!((0.0..=1.0).contains(&b.smoothed_wr));
        }
    }

    // ---- 模式维度（克制矩阵）----
    let mut with_matchups = 0usize;
    let mut total_mode_brawlers = 0usize;
    for mode in &out.modes {
        assert!(mode.total_games > 0);
        for b in &mode.brawlers {
            total_mode_brawlers += 1;
            if !b.matchups.is_empty() {
                with_matchups += 1;
            }
            for m in &b.matchups {
                assert!(m.games >= 6, "低于阈值的对位必须被剔除");
                assert!((0.0..=1.0).contains(&m.win_rate));
                assert!((0.0..=1.0).contains(&m.smoothed_wr));
                assert_ne!(m.opponent_id, b.id, "不应出现自我对位");
            }
        }
    }
    let coverage = with_matchups as f64 / total_mode_brawlers as f64;
    assert!(
        coverage > 0.5,
        "模式级聚合后应有过半英雄具备克制数据，实际 {:.1}%",
        coverage * 100.0
    );

    // ---- 顺位分布 ----
    let mut first = 0usize;
    let mut last = 0usize;
    let mut all = 0usize;
    for map in &out.maps {
        for b in &map.brawlers {
            all += 1;
            match b.pick_order {
                PickOrder::First => first += 1,
                PickOrder::Last => last += 1,
                PickOrder::Middle => {}
            }
        }
    }
    let first_ratio = first as f64 / all as f64;
    let last_ratio = last as f64 / all as f64;
    assert!(
        (0.18..=0.26).contains(&first_ratio),
        "FIRST 占比应约 22%，实际 {:.1}%",
        first_ratio * 100.0
    );
    assert!(
        last_ratio <= 0.26,
        "LAST 占比不应超过 26%，实际 {:.1}%",
        last_ratio * 100.0
    );

    // JSON 序列化 / 反序列化闭环
    let json = out.to_json(false).expect("应能序列化");
    assert!(json.len() < 4_000_000, "单文件体积应控制在 4MB 内");
    let back: MetaOutput = serde_json::from_str(&json).expect("导出的 JSON 必须能被自身解析");
    assert_eq!(back.total_battles, 2000);
    assert_eq!(back.maps.len(), out.maps.len());
    assert_eq!(back.modes.len(), out.modes.len());
    assert_eq!(back.schema_version, out.schema_version);
    assert!(out.to_json(true).unwrap().len() > json.len());
}
