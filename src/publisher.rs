//! 静态 JSON 写入与 Git 自动发布。
//!
//! 设计要点：
//! * **原子写入**：先写 `.tmp` 再 rename，避免前端在 CI 拉取到半个文件；
//! * **幂等推送**：文件无变更时直接优雅退出，返回 `Ok(false)`，不产生空提交。

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use tracing::{info, warn};

use crate::config::Config;
use crate::models::output::MetaOutput;

/// 写入结果。
#[derive(Debug, Clone)]
pub struct WriteReport {
    pub path: PathBuf,
    pub bytes: u64,
}

/// 把分析结果原子写入目标路径。
pub fn write_meta(path: &Path, meta: &MetaOutput, pretty: bool) -> Result<WriteReport> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("创建输出目录 {} 失败", parent.display()))?;
        }
    }

    let json = meta.to_json(pretty)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json.as_bytes())
        .with_context(|| format!("写入临时文件 {} 失败", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("替换目标文件 {} 失败", path.display()))?;

    let bytes = json.len() as u64;
    info!(path = %path.display(), bytes, "导出完成");
    Ok(WriteReport {
        path: path.to_path_buf(),
        bytes,
    })
}

/// 提交并推送导出的 JSON。无变更时返回 `Ok(false)`。
pub fn push(cfg: &Config, path: &Path, message: &str) -> Result<bool> {
    ensure_git_repo()?;

    let path_str = path.to_string_lossy().to_string();
    let pending = run_git(&["status", "--porcelain", "--", &path_str])?;
    if pending.trim().is_empty() {
        info!(path = %path.display(), "文件无变更，跳过提交");
        return Ok(false);
    }

    run_git(&["add", "--", &path_str])?;
    match run_git(&["commit", "-m", message, "--", &path_str]) {
        Ok(_) => {}
        Err(err) => {
            // 提交失败通常是 git 身份未配置或 hook 拒绝，这里如实报错而不是假装成功
            warn!(error = %err, "git commit 失败");
            return Err(err);
        }
    }
    run_git(&["push", &cfg.git_remote, &cfg.git_branch])?;

    info!(remote = %cfg.git_remote, branch = %cfg.git_branch, "已推送到远端");
    Ok(true)
}

/// 确认当前目录是 git 仓库，否则给出可操作提示。
fn ensure_git_repo() -> Result<()> {
    let out = Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .context("执行 git 失败（请确认已安装 git 并在 PATH 中）")?;
    if !out.status.success() {
        bail!(
            "当前目录不是 git 仓库，无法推送：{}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

fn run_git(args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .output()
        .with_context(|| format!("执行 git {} 失败", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} 失败：{}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::output::{
        AnalysisParams, BrawlerRef, MapBrawlerStat, MapMeta, Matchup, ModeBrawlerStat, ModeMeta,
        PickOrder, RolesConfig,
    };

    fn sample_meta() -> MetaOutput {
        MetaOutput {
            schema_version: 2,
            updated_at: "2026-11-21T00:00:00Z".to_string(),
            season_start: "2026-07-20".to_string(),
            total_battles: 42,
            min_tier: 20,
            params: AnalysisParams {
                bayes_prior_weight: 50.0,
                matchup_prior_weight: 6.0,
                matchup_min_samples: 6,
                decay_half_life_days: 30.0,
            },
            brawlers: vec![
                BrawlerRef {
                    id: 16_000_000,
                    name: "Shelly".to_string(),
                },
                BrawlerRef {
                    id: 16_000_001,
                    name: "Colt".to_string(),
                },
            ],
            maps: vec![MapMeta {
                map_name: "Center Stage".to_string(),
                mode: "brawlBall".to_string(),
                total_games: 42,
                brawlers: vec![MapBrawlerStat {
                    id: 16_000_000,
                    name: "Shelly".to_string(),
                    games: 42,
                    wins: 25,
                    win_rate: 0.5952,
                    smoothed_wr: 0.5435,
                    use_rate: 1.0,
                    score: 100.0,
                    pick_order: PickOrder::First,
                }],
            }],
            modes: vec![ModeMeta {
                mode: "brawlBall".to_string(),
                total_games: 42,
                brawlers: vec![ModeBrawlerStat {
                    id: 16_000_000,
                    name: "Shelly".to_string(),
                    games: 42,
                    wins: 25,
                    win_rate: 0.5952,
                    smoothed_wr: 0.5435,
                    use_rate: 1.0,
                    matchups: vec![Matchup {
                        opponent_id: 16_000_001,
                        games: 8,
                        win_rate: 0.625,
                        smoothed_wr: 0.5714,
                    }],
                }],
            }],
            roles: RolesConfig {
                version: 1,
                roles: vec![],
            },
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("bdc-{tag}-{}-{unique}", std::process::id()))
    }

    #[test]
    fn writes_atomic_json_and_creates_parent_dirs() {
        let dir = temp_dir("write");
        let path = dir.join("dist/brawl_meta.json");

        let report = write_meta(&path, &sample_meta(), false).expect("写入应成功");
        assert_eq!(report.path, path);
        assert!(report.bytes > 0);
        assert!(path.exists());
        // 临时文件不应残留
        assert!(!path.with_extension("json.tmp").exists());

        let raw = std::fs::read_to_string(&path).unwrap();
        let back: MetaOutput = serde_json::from_str(&raw).expect("导出的 JSON 必须可解析");
        assert_eq!(back.total_battles, 42);
        assert_eq!(back.maps[0].brawlers[0].pick_order, PickOrder::First);

        // 覆盖写：第二次写入不应报错
        write_meta(&path, &sample_meta(), true).expect("覆盖写应成功");
        assert!(std::fs::read_to_string(&path).unwrap().contains("\n  \"schema_version\""));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compact_output_is_smaller_than_pretty() {
        let meta = sample_meta();
        let compact = meta.to_json(false).unwrap();
        let pretty = meta.to_json(true).unwrap();
        assert!(compact.len() < pretty.len());
        assert!(!compact.contains('\n'));
    }
}
