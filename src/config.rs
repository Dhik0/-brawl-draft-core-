//! 运行配置加载：`.env` + 环境变量 -> 强类型 `Config`。
//!
//! 所有参数都有安全默认值，只有真正调用需要 API Key 的子命令时才会校验密钥。

use std::env;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};

/// 官方直连地址。要求把本机出口 IP 加入白名单，出口 IP 一变就会返回
/// `403 Invalid IP`，因此只作为备选。
pub const OFFICIAL_API_BASE_URL: &str = "https://api.brawlstars.com/v1";

/// 默认 API 根地址：RoyaleAPI 社区中转代理。
///
/// 中转侧用固定出口访问官方接口，本地网络出口 IP 频繁变动也不会再触发
/// `403 Invalid IP`；但访问中转域名本身在国内仍需要本地代理节点（`PROXY_URL`）。
pub const DEFAULT_API_BASE_URL: &str = "https://bsproxy.royaleapi.dev/v1";

/// 引擎全局配置。
#[derive(Debug, Clone)]
pub struct Config {
    /// Supercell API Key（JWT）。`crawl` 子命令必需。
    pub api_key: Option<String>,
    /// 可选出站代理，支持 `http://` 与 `socks5://`。
    pub proxy_url: Option<String>,
    /// API 根地址（默认走 RoyaleAPI 中转代理），便于在测试中指向本地 mock server。
    /// 已去除末尾斜杠，可直接与 `/players/...` 拼接。
    pub api_base_url: String,
    /// 单轮抓取最多处理的玩家数。
    pub crawl_limit: usize,
    /// 异步并发上限。
    pub concurrency: usize,
    /// 429 / 5xx 的最大重试次数。
    pub max_retries: u32,
    /// 赛季重置时间，早于该时间的对局全部丢弃。
    pub season_start: DateTime<Utc>,
    /// 赛季起始日期的原始字符串，用于写入导出元数据。
    pub season_start_raw: String,
    /// 段位阈值（对局 6 人平均 rank）。
    pub min_tier: i64,
    /// 贝叶斯平滑先验权重 m。
    pub bayes_prior_weight: f64,
    /// 1v1 克制矩阵最小交锋样本量。
    pub matchup_min_samples: u32,
    /// SQLite 数据库文件路径。
    pub database_path: PathBuf,
    /// 导出的 JSON 路径。
    pub dist_path: PathBuf,
    /// 英雄定位配置路径。
    pub roles_path: PathBuf,
    /// Git 远端名。
    pub git_remote: String,
    /// Git 分支名。
    pub git_branch: String,
}

impl Config {
    /// 从进程环境（并尝试加载当前目录 `.env`）构建配置。
    pub fn from_env() -> Result<Self> {
        // 缺少 .env 不算错误：CI 里通常直接注入环境变量。
        let _ = dotenvy::dotenv();

        let season_start_raw = env_or("SEASON_START_DATE", "2026-09-01");
        let season_start = parse_season_start(&season_start_raw)?;

        Ok(Self {
            api_key: env_opt("BRAWL_API_KEY"),
            proxy_url: env_opt("PROXY_URL"),
            api_base_url: resolve_api_base_url(env_opt("BRAWL_API_BASE_URL").as_deref()),
            crawl_limit: env_parse("CRAWL_LIMIT", 200usize)?,
            concurrency: env_parse("CRAWL_CONCURRENCY", 10usize)?.max(1),
            max_retries: env_parse("CRAWL_MAX_RETRIES", 3u32)?,
            season_start,
            season_start_raw,
            min_tier: env_parse("MIN_TIER", 20i64)?,
            bayes_prior_weight: env_parse("BAYES_PRIOR_WEIGHT", 50.0f64)?,
            matchup_min_samples: env_parse("MATCHUP_MIN_SAMPLES", 6u32)?,
            database_path: PathBuf::from(env_or("DATABASE_PATH", "brawl.db")),
            dist_path: PathBuf::from(env_or("DIST_PATH", "dist/brawl_meta.json")),
            roles_path: PathBuf::from(env_or("ROLES_PATH", "config/brawler_roles.json")),
            git_remote: env_or("GIT_REMOTE", "origin"),
            git_branch: env_or("GIT_BRANCH", "main"),
        })
    }

    /// 取出 API Key，缺失时给出可操作的错误提示。
    pub fn require_api_key(&self) -> Result<&str> {
        match self.api_key.as_deref() {
            Some(key) if !key.trim().is_empty() => Ok(key.trim()),
            _ => anyhow::bail!(
                "缺少 BRAWL_API_KEY，请在 .env 中填写（或改用 `mock` 子命令离线跑通链路）"
            ),
        }
    }

    /// 赛季起始日期（`YYYY-MM-DD`），用于导出元数据。
    pub fn season_start_date(&self) -> &str {
        &self.season_start_raw
    }
}

/// 解析 API 根地址。
///
/// * 未配置 / 空字符串 -> 回退到 RoyaleAPI 中转代理；
/// * 统一去掉末尾斜杠，避免拼接出 `//players/...` 这种双斜杠路径。
fn resolve_api_base_url(raw: Option<&str>) -> String {
    raw.map(str::trim)
        .filter(|v| !v.is_empty())
        .unwrap_or(DEFAULT_API_BASE_URL)
        .trim_end_matches('/')
        .to_string()
}

/// 解析 `YYYY-MM-DD` 形式的赛季起始日，视为当日 00:00:00 UTC。
fn parse_season_start(raw: &str) -> Result<DateTime<Utc>> {
    let date = NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d")
        .with_context(|| format!("SEASON_START_DATE 格式应为 YYYY-MM-DD，当前为 `{raw}`"))?;
    let naive = date
        .and_hms_opt(0, 0, 0)
        .context("无法构造赛季起始时间")?;
    Ok(naive.and_utc())
}

fn env_opt(key: &str) -> Option<String> {
    env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn env_or(key: &str, default: &str) -> String {
    env_opt(key).unwrap_or_else(|| default.to_string())
}

fn env_parse<T>(key: &str, default: T) -> Result<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    match env_opt(key) {
        None => Ok(default),
        Some(raw) => raw
            .trim()
            .parse::<T>()
            .map_err(|e| anyhow::anyhow!("环境变量 {key} 解析失败（`{raw}`）：{e}")),
    }
}

#[cfg(test)]
impl Config {
    /// 测试用配置：不读环境变量，避免并行测试互相干扰。
    pub fn test_default() -> Self {
        Self {
            api_key: None,
            proxy_url: None,
            api_base_url: DEFAULT_API_BASE_URL.to_string(),
            crawl_limit: 200,
            concurrency: 10,
            max_retries: 3,
            season_start: parse_season_start("2026-09-01").expect("固定日期必然合法"),
            season_start_raw: "2026-09-01".to_string(),
            min_tier: 20,
            bayes_prior_weight: 50.0,
            matchup_min_samples: 6,
            database_path: PathBuf::from("brawl.db"),
            dist_path: PathBuf::from("dist/brawl_meta.json"),
            roles_path: PathBuf::from("config/brawler_roles.json"),
            git_remote: "origin".to_string(),
            git_branch: "main".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn season_start_is_utc_midnight() {
        let dt = parse_season_start("2026-09-01").expect("合法日期应解析成功");
        assert_eq!(dt.to_rfc3339(), "2026-09-01T00:00:00+00:00");
    }

    #[test]
    fn invalid_season_start_is_rejected() {
        assert!(parse_season_start("2026/09/01").is_err());
        assert!(parse_season_start("not-a-date").is_err());
    }

    #[test]
    fn env_parse_falls_back_to_default() {
        let v: usize = env_parse("BRAWL_TEST_DEFINITELY_UNSET_KEY", 42).unwrap();
        assert_eq!(v, 42);
    }

    #[test]
    fn api_base_url_defaults_to_royaleapi_proxy() {
        // 未配置 BRAWL_API_BASE_URL 时必须走中转代理，绕开 403 Invalid IP
        assert_eq!(
            resolve_api_base_url(None),
            "https://bsproxy.royaleapi.dev/v1"
        );
        // 配置了空值 / 纯空白同样回退默认
        assert_eq!(resolve_api_base_url(Some("")), DEFAULT_API_BASE_URL);
        assert_eq!(resolve_api_base_url(Some("   ")), DEFAULT_API_BASE_URL);
    }

    #[test]
    fn api_base_url_can_be_overridden_and_trailing_slash_trimmed() {
        // 显式配置官方直连地址（需要固定出口 IP）
        assert_eq!(
            resolve_api_base_url(Some(OFFICIAL_API_BASE_URL)),
            "https://api.brawlstars.com/v1"
        );
        // 末尾斜杠 / 多余空白必须被清理，避免拼出 //players/...
        assert_eq!(
            resolve_api_base_url(Some(" https://api.brawlstars.com/v1/// ")),
            "https://api.brawlstars.com/v1"
        );
        assert_eq!(
            resolve_api_base_url(Some("http://127.0.0.1:8080/v1/")),
            "http://127.0.0.1:8080/v1"
        );
    }

    #[test]
    fn env_var_wires_into_config() {
        // 该用例会读写进程环境变量，加锁避免与其它用例互相干扰
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // 缺省：不设变量时走中转代理
        std::env::remove_var("BRAWL_API_BASE_URL");
        let default_cfg = Config::from_env().expect("默认配置应加载成功");
        assert_eq!(default_cfg.api_base_url, DEFAULT_API_BASE_URL);

        // 覆盖：自定义地址（含末尾斜杠）应被采纳并规范化
        std::env::set_var("BRAWL_API_BASE_URL", "http://localhost:9999/v1/");
        let custom_cfg = Config::from_env().expect("自定义配置应加载成功");
        std::env::remove_var("BRAWL_API_BASE_URL");

        assert_eq!(custom_cfg.api_base_url, "http://localhost:9999/v1");
    }
}
