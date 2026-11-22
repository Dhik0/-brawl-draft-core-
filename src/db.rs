//! SQLite 存储层：建表、玩家队列调度、对局清洗与哈希去重入库。
//!
//! 去重是整条链路的核心：同一场 3v3 对局会被 6 名玩家的 battlelog 各抓到一次，
//! 因此 `battle_hash = SHA256(battle_time | 排序后的 6 个 tag)` 必须与玩家顺序无关。

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDateTime, Utc};
use rusqlite::{params, Connection, Row};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use crate::config::Config;
use crate::models::api::BattleItem;

/// 建表语句。`brawlers` 是辅助字典表：`battles` 里只存整数 ID，
/// 导出时靠它把 ID 还原成名字。
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS players (
    tag             TEXT PRIMARY KEY,
    tier            INTEGER NOT NULL DEFAULT 0,
    last_crawled_at INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS battles (
    battle_hash TEXT PRIMARY KEY,
    battle_time TEXT NOT NULL,
    mode        TEXT NOT NULL,
    map_name    TEXT NOT NULL,
    avg_tier    INTEGER NOT NULL,
    winners_json TEXT NOT NULL,
    losers_json  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS brawlers (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_players_last_crawled ON players (last_crawled_at);
CREATE INDEX IF NOT EXISTS idx_battles_map ON battles (map_name);
"#;

/// 入库过滤参数。
#[derive(Debug, Clone, Copy)]
pub struct IngestFilter {
    /// 对局 6 人平均段位下限（1~22，16 = 传奇 I）。
    pub min_tier: i64,
    /// 雪球扩充时新玩家的入队门槛（1~22，15 = 神话 III）。
    /// 低于该段位的玩家不会被加入抓取队列，避免队列被低分段稀释。
    pub min_enqueue_tier: i64,
    pub season_start: DateTime<Utc>,
}

impl From<&Config> for IngestFilter {
    fn from(cfg: &Config) -> Self {
        Self {
            min_tier: cfg.min_tier,
            min_enqueue_tier: cfg.min_enqueue_tier,
            season_start: cfg.season_start,
        }
    }
}

/// 一条已清洗完成、可直接落库的对局。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BattleRow {
    pub battle_hash: String,
    pub battle_time: String,
    pub mode: String,
    pub map_name: String,
    pub avg_tier: i64,
    pub winners: Vec<i64>,
    pub losers: Vec<i64>,
}

/// 被丢弃的对局原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterReason {
    /// 非 3v3 排位赛（友谊赛、人机、吃鸡等）
    NotRanked,
    /// 平局
    Draw,
    /// 早于赛季重置时间
    BeforeSeason,
    /// 平均段位低于阈值
    BelowTier,
    /// 字段缺失或结构异常
    Malformed(&'static str),
}

impl FilterReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FilterReason::NotRanked => "not_ranked",
            FilterReason::Draw => "draw",
            FilterReason::BeforeSeason => "before_season",
            FilterReason::BelowTier => "below_tier",
            FilterReason::Malformed(why) => why,
        }
    }
}

/// 单条对局的入库结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestOutcome {
    /// 新对局已写入，附带本局出现的 6 名玩家 Tag（用于滚雪球扩充）
    Inserted(Vec<String>),
    /// 哈希已存在，跳过
    Duplicate,
    /// 未通过清洗规则
    Filtered(FilterReason),
}

/// 数据库统计信息。
#[derive(Debug, Clone, Copy, Default)]
pub struct DbStats {
    pub players: i64,
    pub pending_players: i64,
    pub battles: i64,
    pub maps: i64,
}

/// SQLite 连接封装。
pub struct Db {
    conn: Connection,
}

impl Db {
    /// 打开（并按需创建）数据库文件。
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("创建数据库目录 {} 失败", parent.display()))?;
            }
        }
        let conn = Connection::open(path)
            .with_context(|| format!("打开数据库 {} 失败", path.display()))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;",
        )
        .context("设置 SQLite PRAGMA 失败")?;
        Self::init(conn)
    }

    /// 内存数据库，供测试与 dry-run 使用。
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("创建内存数据库失败")?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.busy_timeout(Duration::from_secs(5))
            .context("设置 busy_timeout 失败")?;
        conn.execute_batch(SCHEMA).context("初始化表结构失败")?;
        Ok(Self { conn })
    }

    /// 直接访问底层连接（测试用）。
    #[cfg(test)]
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    // -----------------------------------------------------------------------
    // 玩家队列
    // -----------------------------------------------------------------------

    /// 插入玩家（已存在则忽略），返回新增条数。
    pub fn enqueue_players<'a, I>(&mut self, tags: I) -> Result<usize>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let tx = self.conn.transaction()?;
        let mut inserted = 0usize;
        {
            let mut stmt = tx.prepare("INSERT OR IGNORE INTO players (tag, tier, last_crawled_at) VALUES (?1, 0, 0)")?;
            for tag in tags {
                let tag = normalize_tag(tag);
                if tag.is_empty() {
                    continue;
                }
                inserted += stmt.execute(params![tag])?;
            }
        }
        tx.commit()?;
        Ok(inserted)
    }

    /// 更新玩家最新排位段位（覆盖写：段位会升降，保留历史最高会让队列失真）。
    ///
    /// **只更新已在队列中的玩家**，不新建行——否则任何被解析出段位的玩家都会
    /// 悄悄混进抓取队列，`min_enqueue_tier` 的门槛就形同虚设。
    pub fn set_player_tier(&mut self, tag: &str, tier: i64) -> Result<()> {
        let tag = normalize_tag(tag);
        self.conn.execute(
            "UPDATE players SET tier = ?2 WHERE tag = ?1",
            params![tag, tier],
        )?;
        Ok(())
    }

    /// 按 `last_crawled_at` 最早优先取出待抓取玩家。
    ///
    /// 段位已知的玩家优先（`tier DESC`）：队列里混着大量 0 段位的待确认玩家时，
    /// 先把已确认的高分段抓完，雪球才不会被低分段稀释。
    pub fn next_crawl_batch(&self, limit: usize) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT tag FROM players ORDER BY tier DESC, last_crawled_at ASC, tag ASC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// 标记玩家最近一次抓取时间（Unix 秒）。
    pub fn mark_crawled(&mut self, tag: &str, ts: i64) -> Result<()> {
        let tag = normalize_tag(tag);
        self.conn.execute(
            "UPDATE players SET last_crawled_at = ?2 WHERE tag = ?1",
            params![tag, ts],
        )?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // 对局清洗与入库
    // -----------------------------------------------------------------------

    /// 清洗并写入一条 battlelog 条目。
    ///
    /// `observer_tag` 是这份 battlelog 的所属玩家：API 只给出「该玩家的胜负」，
    /// 必须靠它判断哪一队是胜方。
    pub fn ingest_battle(
        &mut self,
        item: &BattleItem,
        observer_tag: &str,
        filter: &IngestFilter,
    ) -> Result<IngestOutcome> {
        if !item.is_ranked_3v3() {
            return Ok(IngestOutcome::Filtered(FilterReason::NotRanked));
        }
        if item.is_draw() {
            return Ok(IngestOutcome::Filtered(FilterReason::Draw));
        }

        let teams = match item.teams() {
            Some(t) => t,
            None => return Ok(IngestOutcome::Filtered(FilterReason::Malformed("teams"))),
        };
        if teams.len() != 2 || teams.iter().any(|t| t.len() != 3) {
            return Ok(IngestOutcome::Filtered(FilterReason::Malformed("not_3v3")));
        }

        let battle_time = match parse_battle_time(&item.battle_time) {
            Some(t) => t,
            None => return Ok(IngestOutcome::Filtered(FilterReason::Malformed("battle_time"))),
        };
        if battle_time < filter.season_start {
            return Ok(IngestOutcome::Filtered(FilterReason::BeforeSeason));
        }

        let avg_tier = item.average_tier().unwrap_or(0);
        if avg_tier > 0 && avg_tier < filter.min_tier {
            return Ok(IngestOutcome::Filtered(FilterReason::BelowTier));
        }

        let map_name = match item.event.map.as_deref().map(str::trim) {
            Some(m) if !m.is_empty() => m.to_string(),
            _ => return Ok(IngestOutcome::Filtered(FilterReason::Malformed("map"))),
        };
        // 排位赛的 event.mode 是 soloRanked，真正的玩法在 battle.mode
        let mode = item
            .battle
            .mode
            .as_deref()
            .or(item.event.mode.as_deref())
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .unwrap_or("unknown")
            .to_string();

        let observer = normalize_tag(observer_tag);
        let observer_team = match teams
            .iter()
            .position(|team| team.iter().any(|p| normalize_tag(&p.tag) == observer))
        {
            Some(idx) => idx,
            None => return Ok(IngestOutcome::Filtered(FilterReason::Malformed("observer"))),
        };

        let observer_won = match item.battle.result.as_deref() {
            Some(r) if r.eq_ignore_ascii_case("victory") => true,
            Some(r) if r.eq_ignore_ascii_case("defeat") => false,
            _ => return Ok(IngestOutcome::Filtered(FilterReason::Malformed("result"))),
        };
        let winner_idx = if observer_won {
            observer_team
        } else {
            1 - observer_team
        };

        let mut tags: Vec<String> = Vec::with_capacity(6);
        let mut brawler_names: Vec<(i64, String)> = Vec::new();
        let mut team_ids: Vec<Vec<i64>> = Vec::with_capacity(2);
        // 本局解析出的玩家段位，用于回写 players.tier
        let mut tier_updates: Vec<(String, i64)> = Vec::with_capacity(6);
        // 只有达到入队门槛的玩家才进入雪球队列
        let mut to_enqueue: Vec<String> = Vec::with_capacity(6);

        for team in teams {
            let mut ids = Vec::with_capacity(team.len());
            for p in team {
                if p.brawler.id <= 0 {
                    return Ok(IngestOutcome::Filtered(FilterReason::Malformed("brawler_id")));
                }
                let tag = normalize_tag(&p.tag);
                if tag.is_empty() {
                    return Ok(IngestOutcome::Filtered(FilterReason::Malformed("player_tag")));
                }
                if let Some(tier) = crate::models::api::ranked_tier_of(p) {
                    if tier >= filter.min_enqueue_tier {
                        to_enqueue.push(tag.clone());
                    }
                    tier_updates.push((tag.clone(), tier));
                }
                tags.push(tag);
                ids.push(p.brawler.id);
                if let Some(name) = p.brawler.name.as_deref() {
                    let name = name.trim();
                    if !name.is_empty() {
                        brawler_names.push((p.brawler.id, name.to_string()));
                    }
                }
            }
            team_ids.push(ids);
        }

        let winners = team_ids[winner_idx].clone();
        let losers = team_ids[1 - winner_idx].clone();

        let battle_hash = compute_battle_hash(&item.battle_time, &tags);
        let row = BattleRow {
            battle_hash,
            battle_time: item.battle_time.trim().to_string(),
            mode,
            map_name,
            avg_tier,
            winners,
            losers,
        };

        if self.insert_battle(&row)? {
            self.upsert_brawler_names(&brawler_names)?;
            // 先入队再写段位：set_player_tier 只更新已入队的玩家，
            // 这样未达门槛的玩家不会因为「被解析出段位」而混进队列
            self.enqueue_players(to_enqueue.iter().map(String::as_str))?;
            for (tag, tier) in &tier_updates {
                self.set_player_tier(tag, *tier)?;
            }
            debug!(
                map = %row.map_name,
                hash = %row.battle_hash,
                tier = avg_tier,
                enqueued = to_enqueue.len(),
                "对局入库"
            );
            Ok(IngestOutcome::Inserted(tags))
        } else {
            Ok(IngestOutcome::Duplicate)
        }
    }

    /// 写入一条对局；主键冲突返回 `false`。
    pub fn insert_battle(&mut self, row: &BattleRow) -> Result<bool> {
        let winners_json = serde_json::to_string(&row.winners).context("序列化胜方英雄失败")?;
        let losers_json = serde_json::to_string(&row.losers).context("序列化败方英雄失败")?;
        let changed = self.conn.execute(
            "INSERT OR IGNORE INTO battles
                (battle_hash, battle_time, mode, map_name, avg_tier, winners_json, losers_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                row.battle_hash,
                row.battle_time,
                row.mode,
                row.map_name,
                row.avg_tier,
                winners_json,
                losers_json
            ],
        )?;
        Ok(changed > 0)
    }

    /// 回填英雄名称字典。
    pub fn upsert_brawler_names(&mut self, names: &[(i64, String)]) -> Result<()> {
        if names.is_empty() {
            return Ok(());
        }
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO brawlers (id, name) VALUES (?1, ?2)
                 ON CONFLICT(id) DO UPDATE SET name = excluded.name",
            )?;
            for (id, name) in names {
                stmt.execute(params![id, name])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// 读取英雄字典。
    pub fn brawler_dict(&self) -> Result<HashMap<i64, String>> {
        let mut stmt = self.conn.prepare("SELECT id, name FROM brawlers")?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)))?;
        let mut map = HashMap::new();
        for row in rows {
            let (id, name) = row?;
            map.insert(id, name);
        }
        Ok(map)
    }

    /// 加载全部对局（供分析引擎消费）。
    pub fn load_battles(&self) -> Result<Vec<BattleRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT battle_hash, battle_time, mode, map_name, avg_tier, winners_json, losers_json
             FROM battles",
        )?;
        let rows = stmt.query_map([], row_to_battle)?;

        let mut out = Vec::new();
        for row in rows {
            let raw = row?;
            out.push(raw);
        }
        Ok(out)
    }

    /// 统计信息。
    pub fn stats(&self) -> Result<DbStats> {
        let count = |sql: &str| -> Result<i64> {
            let v: i64 = self.conn.query_row(sql, [], |r| r.get(0))?;
            Ok(v)
        };
        Ok(DbStats {
            players: count("SELECT COUNT(*) FROM players")?,
            pending_players: count("SELECT COUNT(*) FROM players WHERE last_crawled_at = 0")?,
            battles: count("SELECT COUNT(*) FROM battles")?,
            maps: count("SELECT COUNT(DISTINCT map_name) FROM battles")?,
        })
    }

    /// 清空所有业务数据（`mock` 重建场景使用）。
    pub fn clear(&mut self) -> Result<()> {
        self.conn
            .execute_batch("DELETE FROM battles; DELETE FROM brawlers;")
            .context("清空数据失败")?;
        Ok(())
    }
}

/// 从查询结果还原 `BattleRow`（JSON 字段在此阶段保持字符串，稍后统一解析）。
fn row_to_battle(row: &Row<'_>) -> rusqlite::Result<BattleRow> {
    let winners_json: String = row.get(5)?;
    let losers_json: String = row.get(6)?;
    let parse = |raw: &str, col: &str| -> rusqlite::Result<Vec<i64>> {
        serde_json::from_str(raw).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(std::io::Error::other(format!("{col} 解析失败: {e}"))),
            )
        })
    };
    Ok(BattleRow {
        battle_hash: row.get(0)?,
        battle_time: row.get(1)?,
        mode: row.get(2)?,
        map_name: row.get(3)?,
        avg_tier: row.get(4)?,
        winners: parse(&winners_json, "winners_json")?,
        losers: parse(&losers_json, "losers_json")?,
    })
}

/// 规范化玩家 Tag：去空白、统一大写、保证以 `#` 开头。
pub fn normalize_tag(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let body = trimmed.trim_start_matches('#').to_ascii_uppercase();
    format!("#{body}")
}

/// 计算对局去重哈希：`SHA256(battle_time | 字典序排序后的 tag 列表)`。
///
/// 与传入顺序无关，是「同一局被 6 份 battlelog 重复抓到」场景下的唯一去重键。
pub fn compute_battle_hash(battle_time: &str, tags: &[String]) -> String {
    let mut sorted: Vec<String> = tags.iter().map(|t| normalize_tag(t)).collect();
    sorted.sort();
    sorted.dedup();

    let mut hasher = Sha256::new();
    hasher.update(battle_time.trim().as_bytes());
    hasher.update(b"|");
    hasher.update(sorted.join(",").as_bytes());
    hex::encode(hasher.finalize())
}

/// 解析 Supercell 的 `battleTime`（`20260701T123456.000Z`）。
///
/// 注意：末尾的 `Z` 在格式串里是字面量，`DateTime::parse_from_str` 会因缺少时区而失败，
/// 因此先按 `NaiveDateTime` 解析，再显式标记为 UTC。
pub fn parse_battle_time(raw: &str) -> Option<DateTime<Utc>> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    for fmt in ["%Y%m%dT%H%M%S%.3fZ", "%Y%m%dT%H%M%SZ"] {
        if let Ok(ndt) = NaiveDateTime::parse_from_str(raw, fmt) {
            return Some(ndt.and_utc());
        }
    }
    warn!(value = raw, "无法解析 battleTime");
    None
}

/// 把 `battleTime` 格式化成 API 同款字符串（mock 生成器使用）。
pub fn format_battle_time(dt: &DateTime<Utc>) -> String {
    dt.format("%Y%m%dT%H%M%S%.3fZ").to_string()
}

pub mod mock;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::api::{Battle, BattlePlayer, Brawler, Event};

    fn filter() -> IngestFilter {
        IngestFilter {
            min_tier: 16,
            min_enqueue_tier: 15,
            season_start: DateTime::parse_from_rfc3339("2026-09-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        }
    }

    /// 构造一名玩家：`tier` 走排位 battlelog 的 `brawler.trophies` 字段（1~22）。
    fn player(tag: &str, brawler_id: i64, tier: i64) -> BattlePlayer {
        BattlePlayer {
            tag: tag.to_string(),
            name: Some(format!("P{tag}")),
            rank: None,
            brawler: Brawler {
                id: brawler_id,
                name: Some(format!("B{brawler_id}")),
                power: Some(11),
                trophies: Some(tier),
                rank: None,
            },
        }
    }

    /// 构造一局：team_a 胜利（observer 在 team_a）。
    fn battle_item(time: &str, team_a: [&str; 3], team_b: [&str; 3], a_wins: bool) -> BattleItem {
        BattleItem {
            battle_time: time.to_string(),
            event: Event {
                id: Some(1),
                mode: Some("soloRanked".into()),
                map: Some("Center Stage".into()),
            },
            battle: Battle {
                mode: Some("brawlBall".into()),
                battle_type: Some("soloRanked".into()),
                result: Some(if a_wins { "victory" } else { "defeat" }.into()),
                teams: Some(vec![
                    team_a
                        .iter()
                        .enumerate()
                        .map(|(i, t)| player(t, 16000000 + i as i64, 18))
                        .collect(),
                    team_b
                        .iter()
                        .enumerate()
                        .map(|(i, t)| player(t, 16000010 + i as i64, 18))
                        .collect(),
                ]),
                ..Default::default()
            },
            battle_type: None,
        }
    }

    #[test]
    fn battle_hash_is_order_independent() {
        let tags = vec![
            "#AAA".to_string(),
            "#BBB".to_string(),
            "#CCC".to_string(),
            "#DDD".to_string(),
            "#EEE".to_string(),
            "#FFF".to_string(),
        ];
        let mut shuffled = tags.clone();
        shuffled.reverse();
        shuffled.swap(1, 4);

        let h1 = compute_battle_hash("20261012T101112.000Z", &tags);
        let h2 = compute_battle_hash("20261012T101112.000Z", &shuffled);
        assert_eq!(h1, h2, "6 个 tag 顺序不同必须得到同一哈希");
        assert_eq!(h1.len(), 64, "SHA-256 十六进制摘要长度应为 64");
    }

    #[test]
    fn battle_hash_depends_on_time() {
        let tags = vec!["#AAA".to_string(), "#BBB".to_string()];
        assert_ne!(
            compute_battle_hash("20261012T101112.000Z", &tags),
            compute_battle_hash("20261012T101113.000Z", &tags)
        );
    }

    #[test]
    fn tag_normalization_is_case_and_prefix_insensitive() {
        assert_eq!(normalize_tag(" 2pylq "), "#2PYLQ");
        assert_eq!(normalize_tag("#2PYLQ"), "#2PYLQ");
        assert_eq!(normalize_tag(""), "");
    }

    #[test]
    fn duplicate_battle_is_ignored() {
        let mut db = Db::open_in_memory().unwrap();
        let f = filter();
        let item = battle_item(
            "20261012T101112.000Z",
            ["#A1", "#A2", "#A3"],
            ["#B1", "#B2", "#B3"],
            true,
        );

        // 分别以 6 名玩家的视角各入库一次
        for tag in ["#A1", "#A2", "#A3", "#B1", "#B2", "#B3"] {
            let mut item = item.clone();
            if tag.starts_with("#B") {
                item.battle.result = Some("defeat".into());
            }
            let outcome = db.ingest_battle(&item, tag, &f).unwrap();
            if tag == "#A1" {
                assert!(matches!(outcome, IngestOutcome::Inserted(_)));
            } else {
                assert_eq!(outcome, IngestOutcome::Duplicate, "{tag} 视角应判为重复");
            }
        }
        assert_eq!(db.stats().unwrap().battles, 1);
    }

    #[test]
    fn winner_team_is_derived_from_observer_result() {
        let mut db = Db::open_in_memory().unwrap();
        let f = filter();
        // observer 在 B 队且 result=defeat -> 胜方应为 A 队
        let item = battle_item(
            "20261012T101112.000Z",
            ["#A1", "#A2", "#A3"],
            ["#B1", "#B2", "#B3"],
            true,
        );
        let mut seen_from_b = item.clone();
        seen_from_b.battle.result = Some("defeat".into());
        db.ingest_battle(&seen_from_b, "#B1", &f).unwrap();

        let rows = db.load_battles().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].winners, vec![16000000, 16000001, 16000002]);
        assert_eq!(rows[0].losers, vec![16000010, 16000011, 16000012]);
    }

    #[test]
    fn filters_reject_bad_battles() {
        let mut db = Db::open_in_memory().unwrap();
        let f = filter();

        // 平局
        let mut draw = battle_item("20261012T101112.000Z", ["#A1", "#A2", "#A3"], ["#B1", "#B2", "#B3"], true);
        draw.battle.result = Some("draw".into());
        assert_eq!(
            db.ingest_battle(&draw, "#A1", &f).unwrap(),
            IngestOutcome::Filtered(FilterReason::Draw)
        );

        // 友谊赛
        let mut friendly = battle_item("20261012T101112.000Z", ["#A1", "#A2", "#A3"], ["#B1", "#B2", "#B3"], true);
        friendly.battle.battle_type = Some("friendly".into());
        assert_eq!(
            db.ingest_battle(&friendly, "#A1", &f).unwrap(),
            IngestOutcome::Filtered(FilterReason::NotRanked)
        );

        // 赛季前的旧数据
        let old = battle_item("20260101T101112.000Z", ["#A1", "#A2", "#A3"], ["#B1", "#B2", "#B3"], true);
        assert_eq!(
            db.ingest_battle(&old, "#A1", &f).unwrap(),
            IngestOutcome::Filtered(FilterReason::BeforeSeason)
        );

        // 低段位（钻石 I = 10 < 传奇 I = 16）
        let mut low = battle_item("20261012T101112.000Z", ["#A1", "#A2", "#A3"], ["#B1", "#B2", "#B3"], true);
        for team in low.battle.teams.as_mut().unwrap() {
            for p in team {
                p.brawler.trophies = Some(10);
            }
        }
        assert_eq!(
            db.ingest_battle(&low, "#A1", &f).unwrap(),
            IngestOutcome::Filtered(FilterReason::BelowTier)
        );

        assert_eq!(db.stats().unwrap().battles, 0);
    }

    #[test]
    fn inserted_battle_enqueues_all_six_players() {
        let mut db = Db::open_in_memory().unwrap();
        let f = filter();
        let item = battle_item("20261012T101112.000Z", ["#A1", "#A2", "#A3"], ["#B1", "#B2", "#B3"], true);
        db.ingest_battle(&item, "#A1", &f).unwrap();

        let stats = db.stats().unwrap();
        assert_eq!(stats.players, 6, "6 名玩家都应进入抓取队列");
        assert_eq!(stats.pending_players, 6);
        assert_eq!(db.brawler_dict().unwrap().len(), 6);
    }

    #[test]
    fn only_high_tier_players_are_enqueued() {
        let mut db = Db::open_in_memory().unwrap();
        let f = filter();
        // 段位：18 / 15 / 9 / 22 / 14 / 16
        let tiers = [18, 15, 9, 22, 14, 16];
        let mut item = battle_item(
            "20261012T101112.000Z",
            ["#A1", "#A2", "#A3"],
            ["#B1", "#B2", "#B3"],
            true,
        );
        let mut i = 0;
        for team in item.battle.teams.as_mut().unwrap() {
            for p in team {
                p.brawler.trophies = Some(tiers[i]);
                i += 1;
            }
        }
        db.ingest_battle(&item, "#A1", &f).unwrap();

        // 只有 >= 15 的 4 名玩家进入队列（18 / 15 / 22 / 16）
        assert_eq!(db.stats().unwrap().players, 4);
        let mut queued: Vec<String> = db.next_crawl_batch(10).unwrap();
        queued.sort();
        assert_eq!(queued, vec!["#A1", "#A2", "#B1", "#B3"]);
    }

    #[test]
    fn crawl_queue_prefers_higher_tier() {
        let mut db = Db::open_in_memory().unwrap();
        db.enqueue_players(["#LOW", "#MID", "#HIGH"]).unwrap();
        db.set_player_tier("#LOW", 5).unwrap();
        db.set_player_tier("#MID", 15).unwrap();
        db.set_player_tier("#HIGH", 22).unwrap();

        assert_eq!(
            db.next_crawl_batch(3).unwrap(),
            vec!["#HIGH".to_string(), "#MID".to_string(), "#LOW".to_string()]
        );
    }

    #[test]
    fn player_tier_is_overwritten_not_maxed() {
        let mut db = Db::open_in_memory().unwrap();
        db.enqueue_players(["#A"]).unwrap();
        db.set_player_tier("#A", 21).unwrap();
        // 段位会掉，覆盖写才反映「最新」
        db.set_player_tier("#A", 17).unwrap();
        let tier: i64 = db
            .connection()
            .query_row("SELECT tier FROM players WHERE tag = '#A'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(tier, 17);
    }

    #[test]
    fn crawl_queue_prefers_least_recently_crawled() {
        let mut db = Db::open_in_memory().unwrap();
        db.enqueue_players(["#A", "#B", "#C"]).unwrap();
        db.mark_crawled("#A", 1_000).unwrap();
        db.mark_crawled("#B", 500).unwrap();

        let batch = db.next_crawl_batch(2).unwrap();
        assert_eq!(batch, vec!["#C".to_string(), "#B".to_string()]);
    }
}
