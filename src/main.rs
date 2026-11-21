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
    let filter = IngestFilter::from(cfg);

    runtime.block_on(async {
        let crawler = std::sync::Arc::new(Crawler::new(cfg)?);
        if seed {
            crawler.seed_rankings(&mut db, seed_limit).await?;
        }
        let report = crawler.crawl_once(&mut db, limit, &filter).await?;
        let stats = db.stats()?;
        println!(
            "抓取完成：请求 {} / 成功 {} / 新对局 {} / 重复 {} / 过滤 {} / 失败 {}",
            report.requested,
            report.fetched,
            report.inserted,
            report.duplicates,
            report.filtered,
            report.failed
        );
        println!(
            "库内合计：{} 场对局 / {} 张地图 / {} 名玩家（待抓取 {}）",
            stats.battles, stats.maps, stats.players, stats.pending_players
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
        matchup_prior_weight: analyzer::DEFAULT_MATCHUP_PRIOR_WEIGHT,
        matchup_min_samples: cfg.matchup_min_samples,
        decay_half_life_days: analyzer::DEFAULT_DECAY_HALF_LIFE_DAYS,
        min_tier: cfg.min_tier,
        season_start: cfg.season_start_raw.clone(),
        roles,
    };

    let meta = analyze(&battles, &db.brawler_dict()?, &opts);
    let report = publisher::write_meta(&cfg.dist_path, &meta, pretty)?;

    println!(
        "分析完成：{} 场对局 / {} 张地图 / {} 个模式 / {} 名英雄 -> {} ({:.1} KB)",
        meta.total_battles,
        meta.maps.len(),
        meta.modes.len(),
        meta.brawlers.len(),
        report.path.display(),
        report.bytes as f64 / 1024.0
    );

    // 顺位分布：分位分配后应接近 FIRST 22% / LAST 22% / MIDDLE 56%
    let mut first = 0usize;
    let mut middle = 0usize;
    let mut last = 0usize;
    for map in &meta.maps {
        for b in &map.brawlers {
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
            "顺位分布：FIRST {} ({:.1}%) / MIDDLE {} ({:.1}%) / LAST {} ({:.1}%)",
            first,
            first as f64 * 100.0 / total as f64,
            middle,
            middle as f64 * 100.0 / total as f64,
            last,
            last as f64 * 100.0 / total as f64
        );
    }

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
