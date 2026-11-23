//! Tokio 异步滚雪球爬虫。
//!
//! 三条硬约束：
//! 1. **并发限流**：`Semaphore` 控制同时在飞的请求数（默认 10）；
//! 2. **429 / 5xx 指数退避**：最多重试 `CRAWL_MAX_RETRIES` 次，优先遵循 `Retry-After`；
//! 3. **代理兼容**：`PROXY_URL` 支持 `http://` 与 `socks5://`，通过 `reqwest::Proxy::all` 注入。
//!
//! 请求地址完全来自配置（`api_base_url`），默认指向 RoyaleAPI 社区中转代理，
//! 本模块内**不出现任何硬编码的官方域名**。
//!
//! 抓取与入库分离：网络请求全部并发完成后，再串行写 SQLite，
//! 因为 `rusqlite::Connection` 不是 `Sync`，且单写入者能避免锁竞争。
//!
//! 状态推进的口径（决定了队列会不会原地打转）：
//! * **拿到 HTTP 响应就算问过了** —— 2xx（含 `items` 为空、全部被过滤、全部重复）、
//!   404 不存在、其它 4xx 被拒，都必须写回 `last_crawled_at`；
//! * **只有临时失败保留待重试** —— 连接超时、429、5xx 重试耗尽后不写时间戳，
//!   下一轮仍会排到队头；
//! * 401 / 403 属于配置错误（Key 失效 / IP 不在白名单），直接中止整轮，
//!   否则会拿着坏 Key 空转完 2000 个请求。

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use chrono::Utc;
use reqwest::header::{HeaderValue, RETRY_AFTER};
use reqwest::{Client, StatusCode};
use serde::de::DeserializeOwned;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::db::{Db, IngestFilter, IngestOutcome};
use crate::models::api::{BattleLog, RankingList};

/// 单次请求的退避基数（毫秒）。
const BACKOFF_BASE_MS: u64 = 500;
/// 单次退避上限（毫秒）。
const BACKOFF_MAX_MS: u64 = 30_000;
/// 全局排行榜单页上限（Supercell 限制）。
const RANKING_PAGE_LIMIT: usize = 200;

/// 一轮抓取的统计。
#[derive(Debug, Clone, Copy, Default)]
pub struct CrawlReport {
    pub requested: usize,
    /// 拿到 2xx 且解析成功的玩家数（无论最终入库多少条）。
    pub fetched: usize,
    /// 404：玩家不存在，或该玩家没有 battlelog。
    pub not_found: usize,
    /// 其它 4xx：请求被服务端明确拒绝（非法 Tag 等），重试无意义。
    pub rejected: usize,
    pub inserted: usize,
    pub duplicates: usize,
    pub filtered: usize,
    /// 临时失败（超时 / 429 / 5xx），这些玩家保留待重试。
    pub failed: usize,
}

/// 鉴权失败。整轮抓取必须立即中止：拿着坏 Key 继续发请求只会空转。
#[derive(Debug)]
pub struct AuthError(pub String);

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AuthError {}

/// 一次 GET 的结局。
///
/// 只有 `Err` 才算「没问到」；三个 `Ok` 变体都意味着服务端确实回了 HTTP 响应，
/// 调用方必须推进玩家状态。
pub enum FetchOutcome<T> {
    /// 2xx 且响应体可解析。
    Body(T),
    /// 404：玩家不存在，或该玩家没有 battlelog。
    NotFound,
    /// 其它 4xx（400 非法 Tag 等）：已被服务端明确拒绝，重试没有意义。
    Rejected(StatusCode),
}

/// Supercell API 客户端 + 滚雪球调度器。
pub struct Crawler {
    client: Client,
    /// API 根地址，来自 `BRAWL_API_BASE_URL`（已去掉末尾斜杠），
    /// 默认 `https://bsproxy.royaleapi.dev/v1`。
    base: String,
    api_key: String,
    semaphore: Arc<Semaphore>,
    max_retries: u32,
}

impl Crawler {
    /// 依据配置构建爬虫（缺 API Key 会直接报错）。
    pub fn new(cfg: &Config) -> Result<Self> {
        let api_key = cfg.require_api_key()?.to_string();
        Ok(Self {
            client: build_client(cfg)?,
            base: cfg.api_base_url.trim_end_matches('/').to_string(),
            api_key,
            semaphore: Arc::new(Semaphore::new(cfg.concurrency.max(1))),
            max_retries: cfg.max_retries,
        })
    }

    /// 拉取全球排行榜并写入玩家队列。
    pub async fn seed_rankings(&self, db: &mut Db, limit: usize) -> Result<usize> {
        let limit = limit.clamp(1, RANKING_PAGE_LIMIT);
        let path = format!("/rankings/global/players?limit={limit}");
        let list = match self.fetch::<RankingList>(&path).await? {
            FetchOutcome::Body(list) => list,
            FetchOutcome::NotFound => bail!("排行榜接口返回 404：{path}"),
            FetchOutcome::Rejected(status) => {
                bail!("排行榜接口被拒绝：HTTP {status}（{path}）")
            }
        };
        let tags: Vec<String> = list
            .items
            .into_iter()
            .map(|e| e.tag)
            .filter(|t| !t.trim().is_empty())
            .collect();
        let inserted = db.enqueue_players(tags.iter().map(String::as_str))?;
        info!(fetched = tags.len(), inserted, "种子玩家写入完成");
        Ok(inserted)
    }

    /// 执行一轮增量抓取：取本轮到期的 N 名玩家，并发拉 battlelog 后串行入库。
    pub async fn crawl_once(
        self: &Arc<Self>,
        db: &mut Db,
        limit: usize,
        filter: &IngestFilter,
    ) -> Result<CrawlReport> {
        let batch = db.next_crawl_batch(limit)?;
        if batch.is_empty() {
            let stats = db.stats()?;
            if stats.players == 0 {
                info!("玩家队列为空，先执行 seed（排行榜种子）再抓取");
            } else {
                info!(
                    players = stats.players,
                    pending = stats.pending_players,
                    refresh_secs = db.crawl_refresh_secs(),
                    "没有到期玩家：全部仍在刷新间隔内，等待下一轮"
                );
            }
            return Ok(CrawlReport::default());
        }

        let mut report = CrawlReport {
            requested: batch.len(),
            ..Default::default()
        };
        info!(count = batch.len(), base = %self.base, "开始抓取 battlelog");

        // 并发阶段：每个玩家一个任务，实际并发由信号量约束
        let mut set: JoinSet<(String, Result<FetchOutcome<BattleLog>>)> = JoinSet::new();
        for tag in &batch {
            let crawler = Arc::clone(self);
            let tag = tag.clone();
            set.spawn(async move {
                let path = format!("/players/{}/battlelog", encode_tag(&tag));
                let result = crawler.fetch::<BattleLog>(&path).await;
                (tag, result)
            });
        }

        // 串行阶段：写库。now 在本轮开始时取一次，保证同一轮的标记时间可比。
        let now = Utc::now().timestamp();
        let mut fatal: Option<anyhow::Error> = None;

        while let Some(joined) = set.join_next().await {
            let (tag, outcome) = match joined {
                Ok(pair) => pair,
                Err(join_err) => {
                    report.failed += 1;
                    warn!(error = %join_err, "抓取任务异常结束");
                    continue;
                }
            };

            match outcome {
                // 2xx：battlelog 为空、全部被过滤、全部重复都算「问过了」，
                // 否则空号和已扫完的老玩家会一直占着队头。
                Ok(FetchOutcome::Body(log)) => {
                    report.fetched += 1;
                    let mut ingest_ok = true;
                    // battlelog 是「最新在前」，倒序处理让最后写入的 players.tier
                    // 来自该玩家最新的一场对局（set_player_tier 是覆盖写）
                    for item in log.items.iter().rev() {
                        match db.ingest_battle(item, &tag, filter) {
                            Ok(IngestOutcome::Inserted(_)) => report.inserted += 1,
                            Ok(IngestOutcome::Duplicate) => report.duplicates += 1,
                            Ok(IngestOutcome::Filtered(reason)) => {
                                debug!(tag = %tag, reason = reason.as_str(), "对局被过滤");
                                report.filtered += 1;
                            }
                            // 写库失败不是玩家的问题：不推进状态，下轮重试
                            Err(err) => {
                                warn!(tag = %tag, error = %err, "对局入库失败，保留待重试");
                                report.failed += 1;
                                ingest_ok = false;
                            }
                        }
                    }
                    if ingest_ok {
                        db.mark_crawled(&tag, now)?;
                    }
                }
                // 404：玩家不存在或无 battlelog，标记后不再反复重试
                Ok(FetchOutcome::NotFound) => {
                    report.not_found += 1;
                    warn!(tag = %tag, "玩家不存在或无对局记录");
                    db.mark_crawled(&tag, now)?;
                }
                // 其它 4xx：服务端已明确拒绝，重试只会继续失败，同样标记
                Ok(FetchOutcome::Rejected(status)) => {
                    report.rejected += 1;
                    warn!(
                        tag = %tag,
                        status = status.as_u16(),
                        "请求被服务端拒绝（非临时错误），标记为已抓取"
                    );
                    db.mark_crawled(&tag, now)?;
                }
                // 临时失败：不写时间戳，下一轮保留重试
                Err(err) => {
                    if err.downcast_ref::<AuthError>().is_some() {
                        fatal = Some(err);
                        break;
                    }
                    report.failed += 1;
                    warn!(tag = %tag, error = %err, "抓取失败（临时错误，保留待重试）");
                }
            }
        }

        if let Some(err) = fatal {
            return Err(err);
        }

        info!(
            requested = report.requested,
            fetched = report.fetched,
            not_found = report.not_found,
            rejected = report.rejected,
            inserted = report.inserted,
            duplicates = report.duplicates,
            filtered = report.filtered,
            failed = report.failed,
            "本轮抓取完成"
        );
        Ok(report)
    }

    /// 带限流与重试的 GET。
    ///
    /// 返回 `Ok` 表示服务端确实回了 HTTP 响应（调用方必须推进玩家状态）；
    /// `Err` 只用于「没问到」的情况：传输层失败，或 429 / 5xx 重试耗尽。
    /// 401 / 403 以 [`AuthError`] 形式返回，调用方应中止整轮。
    async fn fetch<T>(&self, path: &str) -> Result<FetchOutcome<T>>
    where
        T: DeserializeOwned,
    {
        let url = join_url(&self.base, path);
        let mut last_error: Option<anyhow::Error> = None;

        for attempt in 0..=self.max_retries {
            // permit 在本轮结束时释放，覆盖「发送 + 读取响应体」
            let _permit = self
                .semaphore
                .acquire()
                .await
                .context("并发信号量已关闭")?;

            let response = self.client.get(&url).bearer_auth(&self.api_key).send().await;

            let response = match response {
                Ok(resp) => resp,
                Err(err) => {
                    // 连接层错误（含代理握手失败）同样值得重试
                    if attempt == self.max_retries {
                        return Err(err).with_context(|| format!("请求 {path} 失败（已重试 {attempt} 次）"));
                    }
                    let delay = backoff_delay(attempt, None);
                    warn!(error = %err, attempt = attempt + 1, delay_ms = delay.as_millis(), "网络错误，退避后重试");
                    tokio::time::sleep(delay).await;
                    last_error = Some(err.into());
                    continue;
                }
            };

            let status = response.status();

            if status.is_success() {
                let body = response
                    .json::<T>()
                    .await
                    .with_context(|| format!("解析 {path} 响应失败"))?;
                return Ok(FetchOutcome::Body(body));
            }

            if status == StatusCode::NOT_FOUND {
                return Ok(FetchOutcome::NotFound);
            }

            if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
                return Err(anyhow::Error::new(AuthError(format!(
                    "HTTP {status}：API Key 无效，或当前出口 IP 不在白名单。\
                     直连官方时需把出口 IP 加入白名单（IP 变动即失效），\
                     或改回默认中转代理 BRAWL_API_BASE_URL={}",
                    crate::config::DEFAULT_API_BASE_URL
                ))));
            }

            if status == StatusCode::TOO_MANY_REQUESTS
                || status == StatusCode::REQUEST_TIMEOUT
                || status.is_server_error()
            {
                if attempt == self.max_retries {
                    bail!("{path} 连续 {} 次失败（HTTP {status}）", attempt + 1);
                }
                let delay = backoff_delay(attempt, response.headers().get(RETRY_AFTER));
                warn!(
                    status = status.as_u16(),
                    attempt = attempt + 1,
                    delay_ms = delay.as_millis(),
                    "服务端限流或异常，退避后重试"
                );
                tokio::time::sleep(delay).await;
                last_error = Some(anyhow!("HTTP {status}"));
                continue;
            }

            // 其余 4xx（400 非法 Tag 等）：请求本身被拒绝，重试也不会变好
            return Ok(FetchOutcome::Rejected(status));
        }

        Err(last_error.unwrap_or_else(|| anyhow!("请求 {path} 失败")))
    }
}

/// 归一化代理地址。
///
/// 常见写法有两种：`127.0.0.1:7897`（省略协议）与 `http://127.0.0.1:7897`。
/// 省略协议时按 `http://` 处理——Clash / Mihomo 的 mixed 端口同时支持 HTTP 与 SOCKS，
/// HTTP CONNECT 即可代理 HTTPS 目标，不必强制用户写 `socks5://`。
/// 只有显式写了不支持的协议（如 `ftp://`）才报错。
fn normalize_proxy_url(raw: &str) -> Result<String> {
    let raw = raw.trim();
    const ALLOWED_SCHEMES: [&str; 5] =
        ["http://", "https://", "socks5://", "socks5h://", "socks4://"];

    let lower = raw.to_ascii_lowercase();
    if ALLOWED_SCHEMES.iter().any(|s| lower.starts_with(s)) {
        return Ok(raw.to_string());
    }
    if raw.contains("://") {
        bail!("PROXY_URL 协议不受支持：`{raw}`（仅支持 http:// 与 socks5://）");
    }
    debug!(proxy = raw, "PROXY_URL 未写协议，按 http:// 处理");
    Ok(format!("http://{raw}"))
}

/// 构建 HTTP 客户端：可选出站代理 + 超时 + UA。
pub fn build_client(cfg: &Config) -> Result<Client> {
    let mut builder = Client::builder()
        .user_agent(concat!("brawl-draft-core/", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(30))
        .connect_timeout(Duration::from_secs(10));

    if let Some(proxy) = cfg.proxy_url.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
        let normalized = normalize_proxy_url(proxy)?;
        let proxy = reqwest::Proxy::all(&normalized)
            .with_context(|| format!("PROXY_URL 无效：{proxy}（支持 http:// 与 socks5://）"))?;
        builder = builder.proxy(proxy);
    }

    builder.build().context("构建 HTTP 客户端失败")
}

/// 把 `#TAG` 编码成 URL 安全形式（`%23TAG`）。
pub fn encode_tag(tag: &str) -> String {
    tag.trim().replace('#', "%23")
}

/// 拼接完整 URL。
///
/// `base` 与 `path` 各自的末尾/开头斜杠都被归一化，无论配置里写了
/// `.../v1` 还是 `.../v1/`，都不会拼出 `//players/...` 这种双斜杠路径。
fn join_url(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

/// 指数退避：优先使用 `Retry-After`，否则 500ms 起翻倍，封顶 30s。
fn backoff_delay(attempt: u32, retry_after: Option<&HeaderValue>) -> Duration {
    if let Some(secs) = retry_after
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
    {
        return Duration::from_secs(secs.min(60));
    }
    let factor = 1u64 << attempt.min(6);
    Duration::from_millis(BACKOFF_BASE_MS.saturating_mul(factor).min(BACKOFF_MAX_MS))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_is_url_encoded() {
        assert_eq!(encode_tag("#2PYLQGR"), "%232PYLQGR");
        assert_eq!(encode_tag("  #ABC  "), "%23ABC");
    }

    #[test]
    fn backoff_grows_exponentially_and_caps() {
        assert_eq!(backoff_delay(0, None).as_millis(), 500);
        assert_eq!(backoff_delay(1, None).as_millis(), 1_000);
        assert_eq!(backoff_delay(2, None).as_millis(), 2_000);
        assert_eq!(backoff_delay(3, None).as_millis(), 4_000);
        assert_eq!(backoff_delay(20, None).as_millis(), 30_000);
    }

    #[test]
    fn retry_after_header_wins() {
        let hv = HeaderValue::from_static("7");
        assert_eq!(backoff_delay(0, Some(&hv)).as_secs(), 7);
        // 超过 60s 的 Retry-After 也会被封顶，避免单次睡眠过久
        let huge = HeaderValue::from_static("3600");
        assert_eq!(backoff_delay(0, Some(&huge)).as_secs(), 60);
        // 非法值回退到指数退避
        let bad = HeaderValue::from_static("soon");
        assert_eq!(backoff_delay(1, Some(&bad)).as_millis(), 1_000);
    }

    #[test]
    fn client_accepts_http_and_socks5_proxies() {
        let mut cfg = Config::test_default();
        for proxy in [
            "http://127.0.0.1:8080",
            "socks5://127.0.0.1:1080",
            // 省略协议的常见写法（Clash mixed 端口）
            "127.0.0.1:7897",
        ] {
            cfg.proxy_url = Some(proxy.to_string());
            assert!(build_client(&cfg).is_ok(), "代理 {proxy} 应被接受");
        }

        // 显式写了不支持的协议才报错
        cfg.proxy_url = Some("ftp://127.0.0.1:21".to_string());
        assert!(build_client(&cfg).is_err(), "不支持的协议应报错");

        // 有协议但缺主机，reqwest 解析失败
        cfg.proxy_url = Some("http://".to_string());
        assert!(build_client(&cfg).is_err(), "缺少主机应报错");
    }

    #[test]
    fn proxy_without_scheme_defaults_to_http() {
        assert_eq!(
            normalize_proxy_url("127.0.0.1:7897").unwrap(),
            "http://127.0.0.1:7897"
        );
        assert_eq!(
            normalize_proxy_url("  socks5://1.2.3.4:1080 ").unwrap(),
            "socks5://1.2.3.4:1080"
        );
        assert_eq!(
            normalize_proxy_url("HTTP://proxy.local:8080").unwrap(),
            "HTTP://proxy.local:8080",
            "协议大小写不敏感，原样保留"
        );
        assert!(normalize_proxy_url("ftp://127.0.0.1:21").is_err());
    }

    #[test]
    fn crawler_requires_api_key() {
        let cfg = Config::test_default();
        let err = Crawler::new(&cfg).err().expect("缺少 Key 必须报错");
        assert!(err.to_string().contains("BRAWL_API_KEY"));
    }

    #[test]
    fn crawler_uses_configured_base_url() {
        let mut cfg = Config::test_default();
        cfg.api_key = Some("test-key".to_string());
        cfg.concurrency = 4;
        let crawler = Crawler::new(&cfg).expect("应能构建");
        assert_eq!(crawler.semaphore.available_permits(), 4);
        // 默认走 RoyaleAPI 中转代理，且已去掉末尾斜杠
        assert_eq!(crawler.base, crate::config::DEFAULT_API_BASE_URL);
        assert!(!crawler.base.ends_with('/'));
    }

    #[test]
    fn custom_base_url_is_normalized() {
        let mut cfg = Config::test_default();
        cfg.api_key = Some("test-key".to_string());
        // 用户填了带末尾斜杠的官方直连地址
        cfg.api_base_url = "https://api.brawlstars.com/v1//".to_string();
        let crawler = Crawler::new(&cfg).expect("应能构建");
        assert_eq!(crawler.base, "https://api.brawlstars.com/v1");

        let url = join_url(&crawler.base, "/players/%232PYLQGR/battlelog");
        assert_eq!(
            url,
            "https://api.brawlstars.com/v1/players/%232PYLQGR/battlelog"
        );
        assert!(!url.contains("//players"), "不应出现双斜杠");
    }

    #[test]
    fn join_url_tolerates_slashes_on_both_sides() {
        let expect = "https://bsproxy.royaleapi.dev/v1/rankings/global/players?limit=200";
        for (base, path) in [
            ("https://bsproxy.royaleapi.dev/v1", "/rankings/global/players?limit=200"),
            ("https://bsproxy.royaleapi.dev/v1/", "/rankings/global/players?limit=200"),
            ("https://bsproxy.royaleapi.dev/v1", "rankings/global/players?limit=200"),
            ("https://bsproxy.royaleapi.dev/v1//", "//rankings/global/players?limit=200"),
        ] {
            assert_eq!(join_url(base, path), expect, "base={base} path={path}");
        }
    }

    // -----------------------------------------------------------------------
    // 本地 mock API：验证「拿到 HTTP 响应就必须推进玩家状态」
    // -----------------------------------------------------------------------

    /// 起一个只认路径后缀的极简 HTTP 服务，返回 `http://host:port/v1`。
    async fn spawn_mock_api() -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        const EMPTY_LOG: &str = r#"{"items":[]}"#;
        // 友谊赛：能解析，但会被 IngestFilter 全部过滤
        const FRIENDLY_LOG: &str = r##"{"items":[{
            "battleTime":"20261012T101112.000Z",
            "event":{"id":1,"mode":"brawlBall","map":"Center Stage"},
            "battle":{"mode":"brawlBall","type":"friendly","result":"victory","teams":[
              [{"tag":"#A1","brawler":{"id":16000000,"trophies":18}},
               {"tag":"#A2","brawler":{"id":16000001,"trophies":18}},
               {"tag":"#A3","brawler":{"id":16000002,"trophies":18}}],
              [{"tag":"#B1","brawler":{"id":16000003,"trophies":18}},
               {"tag":"#B2","brawler":{"id":16000004,"trophies":18}},
               {"tag":"#B3","brawler":{"id":16000005,"trophies":18}}]]}}]}"##;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mock API 应能监听本地端口");
        let addr = listener.local_addr().expect("应能取到本地端口");

        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    let n = socket.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let path = req.lines().next().unwrap_or_default().to_string();

                    let (status, body) = if path.contains("%23DEAD") {
                        ("404 Not Found", r#"{"reason":"notFound"}"#)
                    } else if path.contains("%23BOOM") {
                        ("500 Internal Server Error", "{}")
                    } else if path.contains("%23BAD") {
                        ("400 Bad Request", r#"{"reason":"badTag"}"#)
                    } else if path.contains("%23FULL") {
                        ("200 OK", FRIENDLY_LOG)
                    } else {
                        ("200 OK", EMPTY_LOG)
                    };

                    let resp = format!(
                        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\n\
                         content-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                    let _ = socket.shutdown().await;
                });
            }
        });

        format!("http://{addr}/v1")
    }

    fn test_filter() -> IngestFilter {
        IngestFilter {
            min_tier: 16,
            min_enqueue_tier: 15,
            season_start: chrono::DateTime::parse_from_rfc3339("2026-09-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        }
    }

    #[tokio::test]
    async fn every_http_response_advances_player_state() {
        let base = spawn_mock_api().await;
        let mut cfg = Config::test_default();
        cfg.api_key = Some("test-key".to_string());
        cfg.api_base_url = base;
        cfg.max_retries = 0; // 500 立即失败，避免退避拖慢用例
        cfg.concurrency = 4;

        let crawler = Arc::new(Crawler::new(&cfg).expect("应能构建爬虫"));
        let mut db = Db::open_in_memory().unwrap();
        let tags = ["#OK", "#FULL", "#DEAD", "#BAD", "#BOOM"];
        db.enqueue_players(tags).unwrap();
        // 全部设成最高段位：旧实现下这批人会被 tier DESC 永久钉在队头
        for tag in tags {
            db.set_player_tier(tag, 22).unwrap();
        }

        let report = crawler.crawl_once(&mut db, 10, &test_filter()).await.unwrap();
        assert_eq!(report.requested, 5);
        assert_eq!(report.fetched, 2, "200 空日志 + 全被过滤的日志都算 fetched");
        assert_eq!(report.not_found, 1);
        assert_eq!(report.rejected, 1);
        assert_eq!(report.filtered, 1);
        assert_eq!(report.failed, 1);
        assert_eq!(report.inserted, 0);

        let crawled = |db: &Db, tag: &str| -> i64 {
            db.connection()
                .query_row(
                    "SELECT last_crawled_at FROM players WHERE tag = ?1",
                    [tag],
                    |r| r.get(0),
                )
                .unwrap()
        };
        for tag in ["#OK", "#FULL", "#DEAD", "#BAD"] {
            assert!(
                crawled(&db, tag) > 0,
                "{tag} 已拿到 HTTP 响应，必须写回时间戳（含 404 / 4xx / 全过滤 / 空日志）"
            );
        }
        assert_eq!(
            crawled(&db, "#BOOM"),
            0,
            "500 属于临时失败，必须保留待重试，不能写时间戳"
        );

        // 第二轮：只有临时失败的玩家还在队列里
        let report2 = crawler.crawl_once(&mut db, 10, &test_filter()).await.unwrap();
        assert_eq!(report2.requested, 1, "已推进状态的玩家不应再被拉取");
        assert_eq!(report2.failed, 1);
    }

    #[tokio::test]
    async fn auth_failure_aborts_the_round() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let _ = socket.read(&mut buf).await;
                    let _ = socket
                        .write_all(
                            b"HTTP/1.1 401 Unauthorized\r\ncontent-type: application/json\r\n\
                              content-length: 2\r\nconnection: close\r\n\r\n{}",
                        )
                        .await;
                });
            }
        });

        let mut cfg = Config::test_default();
        cfg.api_key = Some("bad-key".to_string());
        cfg.api_base_url = format!("http://{addr}/v1");
        cfg.max_retries = 0;
        let crawler = Arc::new(Crawler::new(&cfg).expect("应能构建爬虫"));

        let mut db = Db::open_in_memory().unwrap();
        db.enqueue_players(["#A", "#B"]).unwrap();

        let err = crawler
            .crawl_once(&mut db, 10, &test_filter())
            .await
            .expect_err("401 必须中止整轮而不是逐个失败");
        assert!(
            err.to_string().contains("API Key"),
            "错误信息应指向鉴权配置，实际：{err}"
        );
        // 鉴权失败不是玩家的问题，不能把队列标记成已抓取
        assert_eq!(db.stats().unwrap().pending_players, 2);
    }
}
