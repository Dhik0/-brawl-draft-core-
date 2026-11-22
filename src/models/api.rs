//! Supercell 官方 API 的**防御性**反序列化结构。
//!
//! 设计原则：
//! * 只有「对局时间 / 阵营成员 / 英雄 ID」这类核心字段是必需的；
//! * 其余字段一律 `Option<T>` 或 `#[serde(default)]`，保证单条脏数据不会让整批解析失败；
//! * 所有结构体派生 `Default`，方便在测试与 mock 生成器中构造。

use serde::Deserialize;

/// 视为排位赛的 `battle.type` 取值。
pub const RANKED_BATTLE_TYPES: [&str; 2] = ["soloRanked", "teamRanked"];

/// 非排位（需要显式剔除）的 `battle.type` 取值。
pub const NON_RANKED_BATTLE_TYPES: [&str; 3] = ["friendly", "challenge", "practice"];

/// 排位段位下界：1 = 青铜 I。
pub const MIN_RANKED_TIER: i64 = 1;
/// 排位段位上界：22 = 职业（Pro）。
pub const MAX_RANKED_TIER: i64 = 22;

// ---------------------------------------------------------------------------
// /rankings/global/players
// ---------------------------------------------------------------------------

/// 全球排行榜响应。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RankingList {
    #[serde(default)]
    pub items: Vec<RankingEntry>,
    #[serde(default)]
    pub paging: Option<Paging>,
}

/// 排行榜单条记录。
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RankingEntry {
    #[serde(default)]
    pub tag: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub trophies: i64,
    #[serde(default)]
    pub rank: Option<i64>,
}

/// 分页游标。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Paging {
    #[serde(default)]
    pub cursors: Option<Cursors>,
}

/// 分页游标明细。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Cursors {
    #[serde(default)]
    pub after: Option<String>,
    #[serde(default)]
    pub before: Option<String>,
}

// ---------------------------------------------------------------------------
// /players/{tag}/battlelog
// ---------------------------------------------------------------------------

/// 玩家战斗日志响应。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct BattleLog {
    #[serde(default)]
    pub items: Vec<BattleItem>,
}

/// 战斗日志中的单场对局。
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BattleItem {
    /// 形如 `20260701T123456.000Z`。
    #[serde(default)]
    pub battle_time: String,
    #[serde(default)]
    pub event: Event,
    #[serde(default)]
    pub battle: Battle,
    /// 部分版本把 type 放在外层。
    #[serde(default, rename = "battleType")]
    pub battle_type: Option<String>,
}

impl BattleItem {
    /// 该局是否属于需要采集的 3v3 排位赛。
    ///
    /// 优先看 `battle.type`，缺失时回退到 `event.mode`，两者都没有则拒绝，
    /// 避免把模式未知的数据混进统计。
    pub fn is_ranked_3v3(&self) -> bool {
        let battle_type = self
            .battle
            .battle_type
            .as_deref()
            .or(self.battle_type.as_deref())
            .unwrap_or_default();

        if NON_RANKED_BATTLE_TYPES
            .iter()
            .any(|t| t.eq_ignore_ascii_case(battle_type))
        {
            return false;
        }

        let ranked = RANKED_BATTLE_TYPES
            .iter()
            .any(|t| t.eq_ignore_ascii_case(battle_type))
            || self
                .event
                .mode
                .as_deref()
                .is_some_and(|m| m.eq_ignore_ascii_case("soloRanked"));

        if !ranked {
            return false;
        }

        // 必须是标准的 3v3 两队结构（单人吃鸡/双人吃鸡在这里被剔除）。
        self.teams().is_some_and(|teams| teams.len() == 2)
    }

    /// 是否平局。
    pub fn is_draw(&self) -> bool {
        self.battle
            .result
            .as_deref()
            .is_some_and(|r| r.eq_ignore_ascii_case("draw"))
    }

    /// 取两队成员；缺失或结构异常时返回 `None`。
    pub fn teams(&self) -> Option<&Vec<Vec<BattlePlayer>>> {
        self.battle
            .teams
            .as_ref()
            .filter(|teams| teams.len() == 2 && teams.iter().all(|t| !t.is_empty()))
    }

    /// 该局出现的全部玩家 Tag（已去重，保持首次出现顺序）。
    pub fn player_tags(&self) -> Vec<String> {
        let mut tags: Vec<String> = Vec::new();
        for team in self.battle.teams.iter().flatten() {
            for p in team {
                let tag = p.tag.trim();
                if !tag.is_empty() && !tags.iter().any(|t| t == tag) {
                    tags.push(tag.to_string());
                }
            }
        }
        tags
    }

    /// 该局全部英雄的 `(id, name)`，用于回填英雄名称字典。
    pub fn brawler_refs(&self) -> Vec<(i64, Option<String>)> {
        self.battle
            .teams
            .iter()
            .flatten()
            .flatten()
            .filter(|p| p.brawler.id > 0)
            .map(|p| (p.brawler.id, p.brawler.name.clone()))
            .collect()
    }

    /// 该局每名玩家的排位段位 `(tag, tier)`；解析失败的玩家被跳过。
    ///
    /// 排位 battlelog 里 `brawler.trophies` 存的**不是**英雄奖杯数，而是官方直接下发的
    /// 排位段位整数（1~22）。范围校验是区分两者的关键：普通对局的真实奖杯是成百上千，
    /// 会被 `1..=22` 天然挡掉。
    pub fn player_tiers(&self) -> Vec<(String, i64)> {
        self.battle
            .teams
            .iter()
            .flatten()
            .flatten()
            .filter_map(|p| {
                let tag = p.tag.trim();
                if tag.is_empty() {
                    return None;
                }
                ranked_tier_of(p).map(|tier| (tag.to_string(), tier))
            })
            .collect()
    }

    /// 6 名玩家的平均排位段位；全员段位不可解析时返回 `None`。
    ///
    /// 取整方式为四舍五入（而非截断），避免 15.6 被记成 15 后跨过门槛。
    pub fn average_tier(&self) -> Option<i64> {
        let tiers = self.player_tiers();
        if tiers.is_empty() {
            return None;
        }
        let sum: i64 = tiers.iter().map(|(_, t)| *t).sum();
        let count = tiers.len() as i64;
        Some((sum + count / 2) / count)
    }
}

/// 解析单个玩家的排位段位（1~22）。
///
/// 主字段是 `brawler.trophies`（排位赛里它是段位整数）；缺失或越界时
/// 再回退到 `brawler.rank` / `player.rank`，同样要求落在 `1..=22`。
pub fn ranked_tier_of(p: &BattlePlayer) -> Option<i64> {
    let valid = |v: &i64| (MIN_RANKED_TIER..=MAX_RANKED_TIER).contains(v);
    p.brawler
        .trophies
        .filter(valid)
        .or_else(|| p.brawler.rank.filter(valid))
        .or_else(|| p.rank.filter(valid))
}

/// 对局事件信息。
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Event {
    #[serde(default)]
    pub id: Option<i64>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub map: Option<String>,
}

/// 对局本体。
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Battle {
    #[serde(default)]
    pub mode: Option<String>,
    /// 对局类型：`soloRanked` / `teamRanked` / `friendly` ...
    #[serde(default, rename = "type")]
    pub battle_type: Option<String>,
    /// `victory` / `defeat` / `draw`
    #[serde(default)]
    pub result: Option<String>,
    #[serde(default)]
    pub rank: Option<i64>,
    #[serde(default)]
    pub trophy_change: Option<i64>,
    #[serde(default)]
    pub duration: Option<i64>,
    #[serde(default)]
    pub star_player: Option<PlayerRef>,
    /// 3v3：两队各若干人；吃鸡模式为 `None`，改用 `players`。
    #[serde(default)]
    pub teams: Option<Vec<Vec<BattlePlayer>>>,
    #[serde(default)]
    pub players: Option<Vec<BattlePlayer>>,
}

/// 仅含 Tag / 名字的轻量引用（`starPlayer` 等）。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PlayerRef {
    #[serde(default)]
    pub tag: String,
    #[serde(default)]
    pub name: Option<String>,
}

/// 对局中的一名玩家。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct BattlePlayer {
    #[serde(default)]
    pub tag: String,
    #[serde(default)]
    pub name: Option<String>,
    /// 玩家在该局的段位（排位赛提供）。
    #[serde(default)]
    pub rank: Option<i64>,
    #[serde(default)]
    pub brawler: Brawler,
}

/// 玩家使用的英雄。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Brawler {
    #[serde(default)]
    pub id: i64,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub power: Option<i64>,
    #[serde(default)]
    pub trophies: Option<i64>,
    /// 排位赛中的英雄段位，用于段位过滤。
    #[serde(default)]
    pub rank: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // 注意：JSON 里含 `"#` 序列（玩家 Tag），所以用 r##"..."## 包裹
    const SAMPLE: &str = r##"{
      "items": [
        {
          "battleTime": "20261012T101112.000Z",
          "event": { "id": 15000001, "mode": "brawlBall", "map": "Center Stage" },
          "battle": {
            "mode": "brawlBall",
            "type": "soloRanked",
            "result": "victory",
            "rank": 1,
            "teams": [
              [
                { "tag": "#AAA", "name": "A", "brawler": { "id": 16000000, "name": "Shelly", "power": 11, "trophies": 18 } },
                { "tag": "#BBB", "name": "B", "brawler": { "id": 16000001, "name": "Colt", "trophies": 16 } },
                { "tag": "#CCC", "name": "C", "brawler": { "id": 16000002, "trophies": 20 } }
              ],
              [
                { "tag": "#DDD", "name": "D", "brawler": { "id": 16000003, "name": "Bull", "trophies": 900 } },
                { "tag": "#EEE", "name": "E", "brawler": { "id": 16000004, "name": "Primo" } },
                { "tag": "#FFF", "name": "F", "brawler": { "id": 16000005, "name": "Barley", "trophies": 0 } }
              ]
            ]
          }
        }
      ]
    }"##;

    #[test]
    fn parses_battlelog_with_partial_fields() {
        let log: BattleLog = serde_json::from_str(SAMPLE).expect("应能解析");
        assert_eq!(log.items.len(), 1);
        let item = &log.items[0];
        assert!(item.is_ranked_3v3());
        assert_eq!(item.teams().map(|t| t.len()), Some(2));
        assert_eq!(item.player_tags().len(), 6);
        assert_eq!(item.event.map.as_deref(), Some("Center Stage"));
        // 只有部分英雄带 name，缺的不能导致解析失败
        assert_eq!(item.brawler_refs().len(), 6);
    }

    #[test]
    fn rejects_non_ranked_and_draws() {
        let mut log: BattleLog = serde_json::from_str(SAMPLE).unwrap();
        let item = &mut log.items[0];

        item.battle.battle_type = Some("friendly".into());
        assert!(!item.is_ranked_3v3());

        item.battle.battle_type = Some("soloRanked".into());
        assert!(item.is_ranked_3v3());

        item.battle.result = Some("draw".into());
        assert!(item.is_draw());
    }

    #[test]
    fn falls_back_to_event_mode_when_type_missing() {
        let mut log: BattleLog = serde_json::from_str(SAMPLE).unwrap();
        let item = &mut log.items[0];
        item.battle.battle_type = None;

        // 排位赛的 event.mode 才是 soloRanked / teamRanked
        item.event.mode = Some("soloRanked".into());
        assert!(item.is_ranked_3v3(), "缺少 battle.type 时应回退到 event.mode");

        // 普通匹配（event.mode 是玩法名）必须被拒绝
        item.event.mode = Some("brawlBall".into());
        assert!(!item.is_ranked_3v3());
    }

    #[test]
    fn trophies_is_parsed_as_ranked_tier() {
        let log: BattleLog = serde_json::from_str(SAMPLE).unwrap();
        let tiers = log.items[0].player_tiers();
        // 18 / 16 / 20 有效；900（真实奖杯）、缺失、0 全部被范围校验挡掉
        assert_eq!(
            tiers,
            vec![
                ("#AAA".to_string(), 18),
                ("#BBB".to_string(), 16),
                ("#CCC".to_string(), 20),
            ]
        );
    }

    #[test]
    fn average_tier_ignores_out_of_range_values() {
        let log: BattleLog = serde_json::from_str(SAMPLE).unwrap();
        // (18 + 16 + 20) / 3 = 18
        assert_eq!(log.items[0].average_tier(), Some(18));
    }

    #[test]
    fn average_tier_is_none_without_valid_tiers() {
        let mut log: BattleLog = serde_json::from_str(SAMPLE).unwrap();
        for team in log.items[0].battle.teams.as_mut().unwrap() {
            for p in team {
                p.brawler.trophies = Some(900); // 普通对局的真实奖杯数
                p.brawler.rank = None;
                p.rank = None;
            }
        }
        assert_eq!(log.items[0].average_tier(), None, "奖杯数不能冒充段位");
    }

    #[test]
    fn average_tier_rounds_half_up() {
        let mut log: BattleLog = serde_json::from_str(SAMPLE).unwrap();
        let teams = log.items[0].battle.teams.as_mut().unwrap();
        // 15,15,16,15,16,16 -> 93/6 = 15.5 -> 16（截断会得到 15，跨不过 Legendary I）
        let vals = [15, 15, 16, 15, 16, 16];
        let mut i = 0;
        for team in teams {
            for p in team {
                p.brawler.trophies = Some(vals[i]);
                i += 1;
            }
        }
        assert_eq!(log.items[0].average_tier(), Some(16));
    }
}
