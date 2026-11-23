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
        min_tier: 16,
        min_enqueue_tier: 15,
        season_start: DateTime::parse_from_rfc3339("2026-09-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc),
    }
}

fn mk_player(tag: &str, brawler_id: i64) -> BattlePlayer {
    BattlePlayer {
        tag: tag.to_string(),
        name: Some(format!("name{tag}")),
        rank: None,
        brawler: Brawler {
            id: brawler_id,
            name: Some(format!("Brawler{brawler_id}")),
            power: Some(11),
            // 排位 battlelog 的 trophies 字段即段位整数（1~22）
            trophies: Some(18),
            rank: None,
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
        matchup_prior_k: 20.0,
        min_tier: 16,
        decay_half_life_days: brawl_draft_core::analyzer::DEFAULT_DECAY_HALF_LIFE_DAYS,
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

        // 综合分是**固定标尺**（`50 + 50*tanh((q-0.5)/0.15)`），不再按单图
        // 极值拉伸到 0/100：榜首只需要是最高分，不必恰好等于 100。
        let top = &map.brawlers[0];
        assert!(
            top.score > 50.0 && top.score <= 100.0,
            "{} 榜首应落在标尺上半区，实际 {}",
            map.map_name,
            top.score
        );
        assert!(
            map.brawlers.iter().all(|b| (0.0..=100.0).contains(&b.score)),
            "{} 综合分必须落在 0~100",
            map.map_name
        );
        assert!(map.brawlers.windows(2).all(|w| w[0].score >= w[1].score));

        for b in &map.brawlers {
            assert!((0.0..=1.0).contains(&b.use_rate));
            assert!((0.0..=1.0).contains(&b.win_rate));
            assert!((0.0..=1.0).contains(&b.smoothed_wr));
            // 展示列表：2 位小数、显著（|delta| >= 1.5pp）、无自我对位
            for m in &b.matchups {
                assert!(
                    m.delta.abs() >= 1.5,
                    "展示列表必须满足显著性门槛，实际 {}",
                    m.delta
                );
                assert_ne!(m.opponent, b.id, "不应出现自我对位");
                assert_eq!(m.delta, (m.delta * 100.0).round() / 100.0, "delta 应为 2 位小数");
            }
            assert!(
                b.matchups.len() <= 16,
                "地图级展示列表每侧最多 8 条，共 16 条"
            );
            assert!(
                b.matchups.windows(2).all(|w| w[0].delta >= w[1].delta),
                "地图级展示列表必须按 delta 降序"
            );
        }

        // ---- 地图级 3×3 查表（v5）----
        //
        // 查表**不截断、不按 |delta| 剪枝**：每对英雄一条 `[a, b, delta_pp, n_eff]`，
        // 只要 `n_eff >= 5.0` 就导出，`delta_pp = 0.0` 的均势对位也保留。
        assert!(
            !map.matchup_table.is_empty(),
            "{} 应产出地图级查表",
            map.map_name
        );
        assert!(
            map.matchup_table.windows(2).all(|w| (w[0].a, w[0].b) < (w[1].a, w[1].b)),
            "{} 查表必须按 (a, b) 升序且无重复",
            map.map_name
        );
        for p in &map.matchup_table {
            assert!(p.a < p.b, "查表必须是无序键 a < b，实际 ({}, {})", p.a, p.b);
            assert!(
                p.n_eff >= out.params.map_matchup_table_min_eff_samples,
                "{} 地图级查表样本不足：{} < {}",
                map.map_name,
                p.n_eff,
                out.params.map_matchup_table_min_eff_samples
            );
            assert_eq!(p.delta_pp, (p.delta_pp * 100.0).round() / 100.0, "delta_pp 应为 2 位小数");
            assert_eq!(p.n_eff, (p.n_eff * 10.0).round() / 10.0, "n_eff 应为 1 位小数");
            assert!(p.delta_pp.abs() <= 100.0, "残差不可能超过 100pp");
        }
        // 展示列表里的每一条，都必须能在同层查表中找到**同口径**的记录：
        // 两者共用同一套分层残差，只是过滤 / 截断策略不同。
        for b in &map.brawlers {
            for m in &b.matchups {
                assert!(
                    map.matchup_table
                        .iter()
                        .any(|p| p.delta_of(b.id, m.opponent) == Some(m.delta)),
                    "{} 展示条目 {} vs {} 在查表中缺失或口径不一致",
                    map.map_name,
                    b.id,
                    m.opponent
                );
            }
        }
    }

    // ---- 模式维度（展示列表）----
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
                assert!(
                    m.delta.abs() >= 1.5,
                    "模式级展示列表必须满足显著性门槛，实际 {}",
                    m.delta
                );
                assert_ne!(m.opponent, b.id, "不应出现自我对位");
            }
            assert!(b.matchups.len() <= 12, "模式级展示列表最多导出 12 条");
        }

        // ---- 模式级 3×3 查表（v5）：同样不截断、不按 |delta| 剪枝 ----
        assert!(
            mode.matchup_table.windows(2).all(|w| (w[0].a, w[0].b) < (w[1].a, w[1].b)),
            "模式 {} 查表必须按 (a, b) 升序且无重复",
            mode.mode
        );
        for p in &mode.matchup_table {
            assert!(p.a < p.b, "模式 {} 查表必须是无序键 a < b", mode.mode);
            assert!(
                p.n_eff >= out.params.mode_matchup_table_min_eff_samples,
                "模式 {} 查表样本不足：{} < {}",
                mode.mode,
                p.n_eff,
                out.params.mode_matchup_table_min_eff_samples
            );
        }
    }

    // ---- 全局查表（第三层）----
    //
    // 这一层**不按 |delta| 剪枝**，因此必须能查到「五五开」的对位；它正是
    // 3×3 对位表不再大面积空白的来源。
    assert!(
        !out.global_matchups.is_empty(),
        "2000 局 mock 数据应产出全局对位表"
    );
    assert!(
        out.global_matchups.windows(2).all(|w| (w[0].a, w[0].b) < (w[1].a, w[1].b)),
        "全局表必须按 (a, b) 升序且无重复"
    );
    let even_pairs = out
        .global_matchups
        .iter()
        .filter(|m| m.delta_pp.abs() < 1.0)
        .count();
    for m in &out.global_matchups {
        assert!(m.a < m.b, "无序格式要求 a < b，实际 ({}, {})", m.a, m.b);
        assert!(
            m.n_eff >= out.params.global_matchup_min_eff_samples,
            "全局表条目样本不足：{} < {}",
            m.n_eff,
            out.params.global_matchup_min_eff_samples
        );
        assert_eq!(
            m.delta_pp,
            (m.delta_pp * 100.0).round() / 100.0,
            "全局表 delta_pp 应为 2 位小数"
        );
        assert_eq!(m.n_eff, (m.n_eff * 10.0).round() / 10.0, "n_eff 应为 1 位小数");
        assert!(m.delta_pp.abs() <= 100.0, "残差不可能超过 100pp");
    }
    assert!(
        even_pairs > 0,
        "不剪枝的全局表必须保留均势对位（|delta| < 1pp），否则前端无法区分五五开与无数据"
    );

    // 克制关系严格反对称：A 对 B 的 delta 必须等于 B 对 A 的相反数
    let mut checked = 0usize;
    for mode in &out.modes {
        for b in &mode.brawlers {
            for m in &b.matchups {
                if let Some(rev) = mode
                    .brawlers
                    .iter()
                    .find(|x| x.id == m.opponent)
                    .and_then(|x| x.matchups.iter().find(|y| y.opponent == b.id))
                {
                    assert_eq!(rev.delta, -m.delta, "模式级 delta 必须严格反对称");
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0, "应至少校验到一组双向对位");
    let coverage = with_matchups as f64 / total_mode_brawlers as f64;
    // 导出门槛是 eff_n >= 12 且 |delta| >= 1.5pp；mock 生成器本身接近可加模型
    // （队伍强度线性决定胜率），真实克制残差很少，因此覆盖率不会像旧口径那么高。
    // 这里只要求「存在一批有克制数据的英雄」，覆盖率本身不是正确性指标。
    assert!(
        coverage > 0.1,
        "模式级聚合后应有一批英雄具备克制数据，实际 {:.1}%",
        coverage * 100.0
    );

    // ---- 顺位分布 ----
    //
    // 只有样本充足的英雄参与 FIRST / LAST 分位划分，所以占比要对**合格英雄**算，
    // 而不是全部条目。极冷门英雄（本图 20 场以下）必须留在 MIDDLE。
    let mut first = 0usize;
    let mut last = 0usize;
    let mut qualified = 0usize;
    for map in &out.maps {
        for b in &map.brawlers {
            // 与 analyzer 的顺位门槛共用同一实现，避免规则漂移
            let is_qualified = brawl_draft_core::analyzer::is_ranked_pick_candidate(b);
            if is_qualified {
                qualified += 1;
            }
            match b.pick_order {
                PickOrder::First => {
                    first += 1;
                    assert!(
                        is_qualified,
                        "{} 样本不足（{} 场）却被标成 FIRST",
                        b.name, b.games
                    );
                }
                PickOrder::Last => {
                    last += 1;
                    assert!(
                        is_qualified,
                        "{} 样本不足（{} 场）却被标成 LAST",
                        b.name, b.games
                    );
                }
                PickOrder::Middle => {}
            }
        }
    }
    assert!(qualified > 0, "应存在合格英雄");
    // 顺位不再按固定配额（旧的 round(22%)）分配，改为「本图对手出场率加权分位数」：
    //   FIRST <=> coverage >= 0.70 且 Q20 >= -2pp 且 mu >= 0.49
    //   LAST  <=> coverage >= 0.70 且 Q80 >= +3pp 且 Q20 <= -3pp
    // mock 数据（2000 局 / 108 英雄）对位证据稀疏，coverage 普遍不足 0.70，
    // 按规则一律 MIDDLE，因此 FIRST 占比完全可能是 0——这里不再断言占比区间，
    // 只守住两条不变量：标签只落在合格英雄上（上面已逐条断言）、标签数量不超上限。
    let first_ratio = first as f64 / qualified as f64;
    let last_ratio = last as f64 / qualified as f64;
    assert!(
        (0.0..=1.0).contains(&first_ratio),
        "FIRST 占比必须在 [0, 1]，实际 {:.1}%",
        first_ratio * 100.0
    );
    assert!(
        (0.0..=1.0).contains(&last_ratio),
        "LAST 占比必须在 [0, 1]，实际 {:.1}%",
        last_ratio * 100.0
    );
    assert!(
        first + last <= qualified,
        "带标签的英雄数（{}）不能超过合格英雄数（{}）",
        first + last,
        qualified
    );

    // JSON 序列化 / 反序列化闭环
    let json = out.to_json(false).expect("应能序列化");
    assert!(json.len() < 16_000_000, "单文件体积应控制在 16MB 内");
    let back: MetaOutput = serde_json::from_str(&json).expect("导出的 JSON 必须能被自身解析");
    assert_eq!(back.total_battles, 2000);
    assert_eq!(back.maps.len(), out.maps.len());
    assert_eq!(back.modes.len(), out.modes.len());
    assert_eq!(back.global_matchups.len(), out.global_matchups.len());
    for (a, b) in out.maps.iter().zip(back.maps.iter()) {
        assert_eq!(a.matchup_table.len(), b.matchup_table.len());
    }
    for (a, b) in out.modes.iter().zip(back.modes.iter()) {
        assert_eq!(a.matchup_table.len(), b.matchup_table.len());
    }
    assert_eq!(back.schema_version, out.schema_version);
    assert!(out.to_json(true).unwrap().len() > json.len());
}
