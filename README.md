# SCOPE v3

SCOPE 是构建在 DESTINY 之上的三级片上缓存行为级 demo。每层都可独立选择 `STT-MRAM`、`SOT-MRAM`、`SRAM`、`Si-eDRAM`、`TFET-eDRAM`、`2D-eDRAM` 或 `OSFET-eDRAM`，并配置容量、相联度、行宽、bank 数和替换策略；不存在固定的“L1=SRAM、L2=eDRAM、L3=MRAM”绑定。

## 1. 运行与目录

```bash
make -j4
make test-scope

# 默认示例：L1 SRAM / L2 TFET-eDRAM / L3 OSFET-eDRAM，Attention
python3 scope.py config/scope_v3.json \
  --json-output results/scope_v3_attention.json

# 同一硬件配置，FFN trace
python3 scope.py config/scope_v3.json --workload ffn \
  --json-output results/scope_v3_ffn.json

# Si-eDRAM 刷新不可行性对照
python3 scope.py config/scope_v3_si_refresh_check.json \
  --json-output results/scope_v3_si_refresh_check.json
```

- `component/`：Destiny 电路组件；根目录只保留 `main.cpp` 和公共头文件。
- `scope/core.py`：多级缓存、功耗、约束和 Guidance；`scope/edram.py`：eDRAM 公式；`scope/openvla_trace.py`：OpenVLA 行为 trace；`scope/cli.py`：命令行入口。
- `config/device_library_v3.json`：v3 单元库；`config/scope_v3.json`：可修改的三级片层示例。

## 2. v3 模型

Destiny 先为每层选择最优 bank/mat/subarray，并输出 `Nrow`、`Ncolumn`、RBL 走线电容与访问晶体管漏端寄生。eDRAM 随后使用：

```text
Lr = C_RBL(Nrow) * ΔV / Ion
Er = 0.5 * C_RBL(Nrow) * [Vdd² - (Vdd-ΔV)²]
Laccess,i = ρr*Lr,i + ρw*Lw,i + Lperipheral,i
Eaccess,i = ρr*Er,i + ρw*Ew,i + Eperipheral,i
```

当前 `Vdd=1.2 V`、`ΔV=0.1 V`；Si/TFET 使用 `Ion,s=33.1 µA/µm`，2D pure 使用 `Ion,d=120 µA/µm`，OSFET pure 使用 `Ion,o=6 µA/µm`。2D 与 OSFET 均明确选择 pure 路径。

只有 Si-eDRAM 计算刷新：`Pref=Nrow*Eref,row/Tref`，其中一行能耗为读+写；刷新串行占用 bank 带宽，访问延迟按可用带宽修正。若刷新占用 `>=100%`，该层判为不可调度。OSFET-eDRAM 的耐久在 v3 中设为不构成约束，允许用于 L3。

命中率不再使用 v2 的指数经验式。v3 根据 OpenVLA-7B 的 Attention/FFN 张量形状生成确定性的 cache-line load/store 序列，再用真实容量、相联度、替换策略逐条执行三级 write-back + write-allocate tag 仿真，因此 `Ri=fR(Ci,W,πi,Ai)`。`ρr/ρw` 也直接来自该序列的读写计数。公开的 VLA-Trace 提供 256 个视觉、32 个文本、7 个动作 token，但不提供特定 GPU 的 ISA 级访存 trace；所以这里是可复现的形状真实行为模型，不冒充硬件 trace。

## 3. 已验证结果与边界

指定映射按等面积密度换算为 32 KiB / 4 MiB / 64 MiB，即 SRAM 的 1×、TFET-eDRAM 的 4×、四层 OSFET-eDRAM 的 16×容量。

| workload | `ρr / ρw` | `R1 / R2 / R3` | 平均延迟 | 平均功耗 |
|---|---:|---:|---:|---:|
| Attention | 0.956745 / 0.043255 | 0.051961 / 0.028093 / 1.000000 | 7.818002 ns | 0.576167 mW |
| FFN | 0.976886 / 0.023114 | 0.002433 / 0.020732 / 1.000000 | 9.193993 ns | 0.766659 mW |

Si-eDRAM 对照的 L1/L2/L3 刷新带宽占用分别为 28.13×、139.57×、382.43×，三层均不可调度；刷新功耗合计 120.452 mW。因此在这组 `Tref=Tret=10 µs` 参数下，Si-eDRAM 不适合 L1/L2/L3。

这是架构探索用行为级 demo，不替代 gem5、真实 GPU trace 或电路签核。外存、NoC、工作频率和 trace 采样范围均为可配置输入。完整数据与运行报告保存在本地 `v3-data.md`、`v3-simulation.md`，并由 Git 忽略。
