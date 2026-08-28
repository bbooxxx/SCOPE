# SCOPE

SCOPE 在原 DESTINY 之上组成可配置的三级片上缓存。每层独立运行一个 DESTINY 实例，L1/L2/L3 均可选择 SRAM、MRAM 或 eDRAM 家族；上层行为模型给出从 load/store 到 L1–L2–L3–片外的端到端延迟、能耗、平均功耗和 FoM。

## 快速运行

```bash
make -j4
make test-scope
python3 scope.py config/scope_example.json --json-output results/scope_example.json
```

当前示例（SRAM / 2D-eDRAM / SOT-MRAM）的结果是 `13.036440 ns`、`396.456403 mW`，全部约束通过。终端会同时打印 L3 load、L2 store 和片外 load 三条具体路径。

`--explore` 已实际验证 27 个 SRAM/SOT-MRAM/2D-eDRAM 三级组合，27 个均完成并通过约束；本次最高 FoM 映射为 SRAM / SOT-MRAM / SOT-MRAM，结果 `12.765098 ns`、`216.892851 mW`，详见 `results/scope_explore.json`。

## 配置

- `config/scope_example.json`：三级容量、相联度、bank、替换策略、BER、刷新、工作负载、crossbar 和片外参数。
- `config/device_library.json`：用户截图中的 7 种器件及全部新指标。
- `config/scope_l{1,2,3}_{sram,edram,mram}.cfg`：每个层级的三种 DESTINY family 配置。
- `results/scope_example.json`：完整 raw/effective 指标、分解和约束结果。

器件与 DESTINY family 必须匹配：SRAM 用 `_sram.cfg`，STT/SOT-MRAM 用 `_mram.cfg`，四种 eDRAM 用 `_edram.cfg`。任何层都可这样更换，并非固定 L1/L2/L3 映射。同一 family 内只需改 `device`；OSFET-eDRAM 还需提供 `bti_endurance_writes_per_line`，且 high-BTI variation 默认不通过，需显式设置 `allow_high_variation=true` 才放行。

```bash
# 指定一条行为路径
python3 scope.py config/scope_example.json --op load --hit-level OFF

# 展开每层 candidates，选择满足约束且 FoM 最高的组合
python3 scope.py config/scope_example.json --explore --json-output results/explore.json
```

命中率支持 `synthetic`、`fixed`、`trace` 三种模式；trace 为每行一个 `{"op":"load|store","address":"0x..."}` 的 JSONL 文件。默认策略为 write-back + write-allocate，填充和脏行写回默认不在需求关键路径，但其能耗会计入平均功耗。

完整方法、公式、参数出处和当前结果见 [建模方案.md](建模方案.md)，实现取舍与验证记录见 [探索过程.md](探索过程.md)。原 DESTINY 说明仍保留在 [README](README)。
