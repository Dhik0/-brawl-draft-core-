//! brawl-draft-core 库入口。
//!
//! 之所以拆出 `lib.rs`：`tests/pipeline_tests.rs` 需要以外部测试的形式复用
//! 去重哈希、解析与统计逻辑，二进制 crate 无法被集成测试导入。

// 分阶段实现期间允许存在尚未接入 CLI 的公共 API，收尾阶段已逐项确认。
pub mod analyzer;
pub mod config;
pub mod crawler;
pub mod db;
pub mod models;
pub mod publisher;
