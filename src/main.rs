//! brawl-draft-core：荒野乱斗排位 BP 数据引擎 CLI。
//!
//! 子命令：
//! * `crawl`   —— 增量抓取高分段玩家对局
//! * `mock`    —— 生成模拟对局（无需 API Key，跑通全链路）
//! * `analyze` —— 聚合统计并导出 `dist/brawl_meta.json`
//! * `push`    —— 提交并推送导出的 JSON

use std::path::Path;
use std::str::FromStr;

use anyhow::{bail, Context, Result};
use brawl_draft_core::analyzer::{self, analyze, AnalyzeOptions};
use brawl_draft_core::config::Config;
use brawl_draft_core::crawler::Crawler;
use brawl_draft_core::db::{mock, Db, IngestFilter};
use brawl_draft_core::models::output::{PickOrder, RolesConfig};
use brawl_draft_core::publisher;
use clap::{Parser, Subcommand};
use tracing::info;
use tracing::level_filters::LevelFilter;

#[derive(Parser, Debug)]
#[command(
    name = "brawl-draft-core",
    version,
    about = "荒野乱斗排位 BP 数据引擎：采集 -> 清洗去重 -> 并行统计 -> 静态 JSON 分发",
    long_about = None
)]
struct Cli {
    /// 子命令
    #[command(subcommand)]
    command: Cmd,

    /// 输出美化（缩进）的 JSON，默认紧凑输出以压缩体积
    #[arg(long, global = true)]
    pretty: bool,

    /// 日志级别：trace / debug / info / warn / error
    #[arg(long, global = true, default_value = "info", value_name = "LEVEL")]
    log: String,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// 执行一轮增量抓取（需要 BRAWL_API_KEY）
    Crawl {
        /// 本轮最多抓取的玩家数，默认取 CRAWL_LIMIT
        #[arg(long, value_name = "N")]
        limit: Option<usize>,
        /// 抓取前先拉取全球排行榜作为种子
        #[arg(long)]
        seed: bool,
        /// 种子排行榜的取数上限（最大 200）
        #[arg(long, default_value_t = 200, value_name = "N")]
        seed_limit: usize,
    },
    /// 生成模拟排位对局（无需 API Key）
    Mock {
        /// 生成的对局数量
        #[arg(long, default_value_t = 1000, value_name = "N")]
        count: usize,
        /// 随机种子，同种子结果可复现
        #[arg(long, default_value_t = 20261121)]
        seed: u64,
        /// 追加到现有对局（默认先清空 battles 表）
        #[arg(long)]
        keep: bool,
    },
    /// 聚合计算并导出 dist/brawl_meta.json
    Analyze,
    /// 提交并推送导出的 JSON 到 Git 远端
    Push {
        /// 自定义提交信息
        #[arg(long, value_name = "MSG")]
        message: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();
    init_tracing(&cli.log);

    let Cli {
        command,
        pretty,
        log: _,
    } = cli;

    if let Err(err) = run(command, pretty) {
        eprintln!("错误: {err:#}");
        std::process::exit(1);
    }
}

fn init_tracing(level: &str) {
    let filter = LevelFilter::from_str(level.trim()).unwrap_or(LevelFilter::INFO);
    tracing_subscriber::fmt()
        .with_max_level(filter)
        .with_target(false)
        .init();
}

fn run(command: Cmd, pretty: bool) -> Result<()> {
    let cfg = Config::from_env()?;
    match command {
        Cmd::Crawl {
            limit,
            seed,
            seed_limit,
        } => cmd_crawl(&cfg, limit.unwrap_or(cfg.crawl_limit), seed, seed_limit),
        Cmd::Mock { count, seed, keep } => cmd_mock(&cfg, count, seed, keep),
        Cmd::Analyze => cmd_analyze(&cfg, pretty),
        Cmd::Push { message } => cmd_push(&cfg, message.as_deref()),
    }
}

// ---------------------------------------------------------------------------
// crawl
// ---------------------------------------------------------------------------

fn cmd_crawl(cfg: &Config, limit: usize, seed: bool, seed_limit: usize) -> Result<()> {
    if limit == 0 {
        bail!("--limit 必须大于 0");
    }

    let runtime = tokio::runtime::Runtime::new().context("创建 Tokio 运行时失败")?;
    let mut db = Db::open(&cfg.database_path)?;
    // 已抓取玩家的重新排队间隔：0 表示每轮都可重抓
    db.set_crawl_refresh_secs(cfg.crawl_refresh_hours.saturating_mul(3600));
    let filter = IngestFilter::from(cfg);

    runtime.block_on(async {
        let crawler = std::sync::Arc::new(Crawler::new(cfg)?);
        if seed {
            crawler.seed_rankings(&mut db, seed_limit).await?;
        }
        let report = crawler.crawl_once(&mut db, limit, &filter).await?;
        let stats = db.stats()?;
        println!(
            "抓取完成：请求 {} / 成功 {} / 不存在 {} / 被拒 {} / 新对局 {} / 重复 {} / 过滤 {} / 失败 {}",
            report.requested,
            report.fetched,
            report.not_found,
            report.rejected,
            report.inserted,
            report.duplicates,
            report.filtered,
            report.failed
        );
        println!(
            "库内合计：{} 场对局 / {} 张地图 / {} 名玩家（从未抓取 {} / 本轮可抓 {} / 刷新间隔 {}h）",
            stats.battles,
            stats.maps,
            stats.players,
            stats.pending_players,
            stats.due_players,
            cfg.crawl_refresh_hours
        );
        Ok::<(), anyhow::Error>(())
    })
}

// ---------------------------------------------------------------------------
// mock
// ---------------------------------------------------------------------------

fn cmd_mock(cfg: &Config, count: usize, seed: u64, keep: bool) -> Result<()> {
    if count == 0 {
        bail!("--count 必须大于 0");
    }
    let mut db = Db::open(&cfg.database_path)?;
    if !keep {
        db.clear()?;
        info!("已清空历史对局与英雄字典（使用 --keep 可追加）");
    }

    let filter = IngestFilter::from(cfg);
    let summary = mock::generate(&mut db, count, seed, &filter)?;
    let stats = db.stats()?;

    println!(
        "模拟数据生成完成：请求 {} / 入库 {} / 重复 {} / 过滤 {}",
        summary.requested, summary.inserted, summary.duplicates, summary.filtered
    );
    println!(
        "库内合计：{} 场对局 / {} 张地图 / {} 名玩家",
        stats.battles, stats.maps, stats.players
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// analyze
// ---------------------------------------------------------------------------

fn cmd_analyze(cfg: &Config, pretty: bool) -> Result<()> {
    let db = Db::open(&cfg.database_path)?;
    let battles = db.load_battles()?;
    if battles.is_empty() {
        bail!(
            "数据库中没有对局，请先执行 `mock --count 5000` 或 `crawl --limit 200 --seed`（当前库：{}）",
            cfg.database_path.display()
        );
    }

    let roles = RolesConfig::load(&cfg.roles_path)?;
    let opts = AnalyzeOptions {
        bayes_prior_weight: cfg.bayes_prior_weight,
        matchup_prior_k: analyzer::MATCHUP_PRIOR_K,
        min_tier: cfg.min_tier,
        decay_half_life_days: analyzer::DEFAULT_DECAY_HALF_LIFE_DAYS,
        season_start: cfg.season_start_raw.clone(),
        roles,
    };

    let meta = analyze(&battles, &db.brawler_dict()?, &opts);
    let report = publisher::write_meta(&cfg.dist_path, &meta, pretty)?;

    print_tier_audit(&meta);

    // BO3 聚簇：同一批 6 人在 10 分钟内连打的小局按 1/k 降权，
    // 因此「原始对局数」与「有效 BP 组数」不是一回事
    let (raw_battles, effective_groups) = analyzer::series_cluster_report(&battles, cfg.min_tier);
    println!(
        "BO3 聚簇：{} 场原始对局 -> {} 个有效 BP 组（同阵容 10 分钟内连打按 1/k 降权，{} 秒窗口）",
        raw_battles,
        effective_groups,
        analyzer::SERIES_CLUSTER_WINDOW_SECS
    );

    println!(
        "分析完成：{} 场对局 / {} 张地图 / {} 个模式 / {} 名英雄 -> {} ({:.2} MB / {} 字节)",
        meta.total_battles,
        meta.maps.len(),
        meta.modes.len(),
        meta.brawlers.len(),
        report.path.display(),
        report.bytes as f64 / (1024.0 * 1024.0),
        report.bytes
    );

    // 单图平均样本量：低于 500 说明过滤口径过严，导出的是稀疏数据
    let avg_per_map = if meta.maps.is_empty() {
        0.0
    } else {
        meta.maps.iter().map(|m| m.total_games).sum::<u64>() as f64 / meta.maps.len() as f64
    };
    println!(
        "单图平均样本：{:.0} 局（{} 张地图，最少 {} 局）",
        avg_per_map,
        meta.maps.len(),
        meta.maps.iter().map(|m| m.total_games).min().unwrap_or(0)
    );

    // 顺位分布：只在**合格英雄**里统计（极冷门英雄不参与分位划分）
    let mut first = 0usize;
    let mut middle = 0usize;
    let mut last = 0usize;
    for map in &meta.maps {
        for b in &map.brawlers {
            if !analyzer::is_ranked_pick_candidate(b) {
                continue;
            }
            match b.pick_order {
                PickOrder::First => first += 1,
                PickOrder::Middle => middle += 1,
                PickOrder::Last => last += 1,
            }
        }
    }
    let total = first + middle + last;
    if total > 0 {
        println!(
            "顺位分布（合格英雄 {} 名）：FIRST {} ({:.1}%) / MIDDLE {} ({:.1}%) / LAST {} ({:.1}%)",
            total,
            first,
            first as f64 * 100.0 / total as f64,
            middle,
            middle as f64 * 100.0 / total as f64,
            last,
            last as f64 * 100.0 / total as f64
        );
    }

    // 克制条目分两套口径，必须分开打印，避免把「查表」和「展示」混为一谈：
    //   * 查表（matchup_table / global_matchups）：无序键、每对一条、不截断、不剪枝
    //   * 展示（brawlers[].matchups）：每英雄 Top-K + 显著性过滤
    let map_table_entries: usize = meta.maps.iter().map(|m| m.matchup_table.len()).sum();
    let mode_table_entries: usize = meta.modes.iter().map(|m| m.matchup_table.len()).sum();
    let global_table_entries = meta.global_matchups.len();

    let even_in = |pairs: &[brawl_draft_core::models::output::MatchupPair]| {
        pairs.iter().filter(|m| m.delta_pp == 0.0).count()
    };
    let map_even: usize = meta.maps.iter().map(|m| even_in(&m.matchup_table)).sum();
    let mode_even: usize = meta.modes.iter().map(|m| even_in(&m.matchup_table)).sum();
    let global_even = even_in(&meta.global_matchups);

    println!("3×3 对位查表（无序键 a<b / 每对一条 / 不截断 / 不按 |delta| 剪枝）：");
    println!(
        "          地图级 {} 条（n_eff >= {:.1}，其中均势 {} 条）",
        map_table_entries, meta.params.map_matchup_table_min_eff_samples, map_even
    );
    println!(
        "          模式级 {} 条（n_eff >= {:.1}，其中均势 {} 条）",
        mode_table_entries, meta.params.mode_matchup_table_min_eff_samples, mode_even
    );
    println!(
        "          全局级 {} 条（n_eff >= {:.1}，其中均势 {} 条）",
        global_table_entries, meta.params.global_matchup_min_eff_samples, global_even
    );

    let map_entries: usize = meta
        .maps
        .iter()
        .map(|m| m.brawlers.iter().map(|b| b.matchups.len()).sum::<usize>())
        .sum();
    let mode_entries: usize = meta
        .modes
        .iter()
        .map(|m| m.brawlers.iter().map(|b| b.matchups.len()).sum::<usize>())
        .sum();
    println!(
        "展示列表（每英雄 Top-K，eff_n >= {:.0}/{:.0} 且 |delta| >= {:.1}pp）：地图级 {} 条 / 模式级 {} 条",
        meta.params.map_matchup_min_samples,
        meta.params.mode_matchup_min_eff_samples,
        meta.params.matchup_min_abs_delta * 100.0,
        map_entries,
        mode_entries
    );
    println!(
        "          收缩强度 K = {:.0}（伪场次）",
        meta.params.matchup_smoothing_c
    );

    for map in meta.maps.iter().take(8) {
        let top: Vec<String> = map
            .brawlers
            .iter()
            .take(3)
            .map(|b| format!("{}({:.1})", b.name, b.score))
            .collect();
        println!(
            "  {:<16} {:<10} {:>5} 局  {}",
            map.map_name,
            map.mode,
            map.total_games,
            top.join("  ")
        );
    }
    Ok(())
}

/// 打印段位分布与过滤结果。
///
/// 按「青铜 / 白银 / 黄金 / 钻石 / 神话 / 传奇 / 大师 / 职业」分组汇总，
/// 再附一行逐段位明细——用于确认 `brawler.trophies` 的解析口径是否正确。
fn print_tier_audit(meta: &brawl_draft_core::models::output::MetaOutput) {
    use brawl_draft_core::models::output::{tier_group, tier_label};

    let audit = &meta.tier_audit;
    println!(
        "段位门槛 MIN_TIER={}（{}）：过滤前 {} 局 / 过滤后 {} 局（段位未知 {} 局）",
        audit.min_tier,
        tier_label(audit.min_tier),
        audit.battles_total,
        audit.battles_kept,
        audit.unknown_tier_battles
    );

    let total = audit.battles_total.max(1) as f64;
    for group in [
        "青铜", "白银", "黄金", "钻石", "神话", "传奇", "大师", "职业", "未知",
    ] {
        let battles: u64 = audit
            .distribution
            .iter()
            .filter(|b| tier_group(b.tier) == group)
            .map(|b| b.battles)
            .sum();
        if battles == 0 {
            continue;
        }
        println!(
            "  {:<4} {:>8} 局  ({:>5.1}%)",
            group,
            battles,
            battles as f64 * 100.0 / total
        );
    }

    // 逐段位明细（降序），最多 22 行
    let mut detail: Vec<&brawl_draft_core::models::output::TierBucket> = audit
        .distribution
        .iter()
        .filter(|b| b.battles > 0)
        .collect();
    detail.sort_by(|a, b| b.tier.cmp(&a.tier));
    if !detail.is_empty() {
        let line: Vec<String> = detail
            .iter()
            .map(|b| format!("{}={}", b.label, b.battles))
            .collect();
        println!("  明细：{}", line.join("  "));
    }

    if !audit.filter_effective() {
        println!(
            "  ⚠ 段位门槛无从判定：{} 局段位字段不可解析（期望 brawler.trophies 落在 1..=22），\
             实际统计的是全部分段",
            audit.unknown_tier_battles
        );
    }
}

// ---------------------------------------------------------------------------
// push
// ---------------------------------------------------------------------------

fn cmd_push(cfg: &Config, message: Option<&str>) -> Result<()> {
    let path = Path::new(&cfg.dist_path);
    if !path.exists() {
        bail!(
            "找不到 {}，请先执行 `analyze` 生成导出文件",
            path.display()
        );
    }

    let default_message = format!(
        "chore(meta): update brawl meta {}",
        chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
    );
    let message = message.unwrap_or(&default_message);

    if publisher::push(cfg, path, message)? {
        println!("已提交并推送到 {}/{}", cfg.git_remote, cfg.git_branch);
    } else {
        println!("导出文件无变更，无需推送");
    }
    Ok(())
}
