# SCOPE v4

SCOPE 是构建在 DESTINY 之上的三级片上缓存行为级模拟器。每层均可独立配置器件、容量、相联度、行宽、bank 数和替换策略；器件库包含 SRAM、STT/SOT-MRAM、Si/TFET/2D/OSFET-eDRAM，不存在固定层级绑定。v4 的默认设计空间经过约束预筛后包含 48 个合理映射，最终由实测 `1/(latency × power)` 选择，而不是硬编码结果。

## 1. 构建与运行

```bash
make -j4
make test-scope

# 48 个映射：OpenVLA Attention
python3 scope.py config/scope_v4.json --explore \
  --json-output results/scope_v4_attention.json

# 同一设计空间：OpenVLA FFN
python3 scope.py config/scope_v4.json --workload ffn --explore \
  --json-output results/scope_v4_ffn.json
```

- `component/`：DESTINY 电路组件；`model/`：C++ 访问生成与三级缓存模拟。
- `scope/`：Python 编排、DESTINY 输出解析、功耗/FoM/约束汇总。
- `config/device_library_v3.json`：来自 `Memory_Tech_Comparison.xlsx` 的单元库；`config/scope_v4.json`：v4 参数与候选空间。

自动搜索排除明显不适用组合：Si-eDRAM 的刷新带宽不可调度，OSFET 的高 BTI variation 不进入 L1/L2，STT-MRAM 不进入延迟敏感候选。它们仍保留在器件库中，可在任意层手动配置并接受同一套约束检查。

TFET-eDRAM 也参与 L1 搜索；配置显式加入 0.25 ns 的 L1 eDRAM 接口/恢复时序开销。该值是可修改的层级参数，不改动工作簿中的器件数据。

## 2. v4 模型

OpenVLA-7B 使用 Llama-2-7B 主干；v4 取一层真实形状：`sequence=295`、`hidden=4096`、`heads=32`、`head_dim=128`、FFN `intermediate=11008`。Attention 按 FlashAttention 的 Q/K/V tile 与在线 softmax 数据流生成访问，FFN 按 CUTLASS 风格的 tile-based SwiGLU GEMM/GEMV 生成访问。C++ 固定随机种子，从对应张量地址域抽取 16-byte load/store，读比例设为 0.75；这是 ISA 粒度的行为样本，不冒充某一 GPU 的硬件 trace。[OpenVLA](https://github.com/openvla/openvla/blob/main/prismatic/conf/models.py)、[FlashAttention](https://github.com/Dao-AILab/flash-attention)、[CUTLASS GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api_3x.md)

命中率不使用容量比例公式，也没有针对目标映射拟合。C++ 对每条访问执行 3 级 set-associative、LRU、write-back + write-allocate 模拟，并分开执行 warmup 与 measured phase；这与 ChampSim 的 trace/warmup 方法及 gem5 默认 set-associative + LRU 口径一致。容量直接改变可驻留 cache line 数，所以 64 MiB OSFET L3 能保留 24/32 MiB 抽样工作集的大部分复用数据；1.5625% 的跨迭代冷流仍会到达片外。[ChampSim](https://github.com/ChampSim/ChampSim)、[gem5 Classic caches](https://www.gem5.org/documentation/general_docs/memory_system/classic_caches/)

片外数据换为 Jetson AGX Orin 对齐的 256-bit LPDDR5-6400、204.8 GB/s。随机闭页访问延迟使用 Ramulator2 的 `nRCD+nCL+nBL+nRPpb` 与 1.25 ns CK，得到 67.5 ns；能耗使用 2.5 pJ/bit，即 64 B 为 1.28 nJ。[NVIDIA Orin Technical Brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)、[Ramulator2](https://github.com/CMU-SAFARI/ramulator2)、[DOE EES2 Roadmap](https://www.energy.gov/documents/energy-efficiency-scaling-two-decades-research-and-development-roadmap-draft)

## 3. 已验证结果

等面积基准为 L1/L2/L3 SRAM `32 KiB / 1 MiB / 4 MiB`。TFET-eDRAM 以 35 F² 对 140 F² 得到 4× 容量；4-tier OSFET-eDRAM 以 `35/4 F²` 得到 16×，因此默认目标为 `32 KiB SRAM / 4 MiB TFET-eDRAM / 64 MiB OSFET-eDRAM`。

| workload | `R1 / R2 / R3` | LPDDR5 到达率 | 平均延迟 | 平均功耗 | `1/(ns·mW)` |
|---|---:|---:|---:|---:|---:|
| Attention | 0.496749 / 0.153608 / 0.964425 | 0.015153 | 6.282637 ns | 1.411824 mW | 0.112740 |
| FFN | 0.498510 / 0.114978 / 0.965962 | 0.015107 | 6.384046 ns | 1.921136 mW | 0.081535 |

两种 workload 中，`SRAM / TFET-eDRAM / OSFET-eDRAM` 均为 48 个可行候选的 FoM 第 1 名。Attention 次优 `TFET-eDRAM / TFET-eDRAM / OSFET-eDRAM` 的 FoM 为 0.109373；FFN 次优为 0.079003。完整 JSON、数据来源和关键点保存在本地并由 Git 忽略。该模型用于架构探索，不替代真实 GPU trace、gem5/Ramulator2 联合仿真或电路签核。
