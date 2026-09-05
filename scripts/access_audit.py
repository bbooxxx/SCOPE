"""Validate and publish the access-accounting correction without changing measurements."""

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v9_reports import main as publish_v9, table, LABELS

CASES = ("all_sram", "sram_mram_mram", "sram_osfet_osfet", "optimized")


def validate(suite):
    assert len(suite["architecture_comparison"]) == 4
    for case, r in suite["case_reports"].items():
        assert r["power_metric"] == "system_average"
        trace = r["hit_rate_model"]["trace_metadata"]
        assert trace["trace_kind"] == "tiled"
        assert trace["sample_accesses"] == trace["trace_cycle_accesses"]
        assert trace["warmup_accesses"] == 0
        assert trace["power_activity_model"] == "legacy_v9_analytical"
        assert trace["power_activity_transactions_per_operator"] == trace["power_activity_counts"]["total"]
        assert math.isclose(r["memory_access_rate_per_s"],
                            trace["power_activity_transactions_per_operator"] * trace["policy_frequency_hz"])
        n = r["conditional_accesses"][0]
        assert sum(r["conditional_hits"])+trace["offchip_loads"]+trace["offchip_stores"] == n
        for i in range(3):
            assert trace["fills"][i] == r["conditional_accesses"][i]-r["conditional_hits"][i]
            assert math.isclose(r["hit_rates"][i], r["conditional_hits"][i]/r["conditional_accesses"][i], abs_tol=1e-14)
        assert math.isclose(sum(r["global_hit_probabilities"])+r["offchip_reach_probability"], 1)
        assert math.isclose(sum(p["probability"] for p in r["request_path_costs"]), 1)
        assert math.isclose(sum(p["probability"]*p["latency_ns"] for p in r["request_path_costs"]), r["average_latency_ns"])
        assert math.isclose(r["single_access_dynamic_power_mw"],
                            1000*r["expected_dynamic_energy_nj_per_request"]/r["serialized_service_time_ns_per_request"])
        assert math.isclose(r["average_power_mw"], r["system_average_power_mw"])
        assert math.isclose(r["dynamic_power_mw"],
                            r["memory_access_rate_per_s"]*r["expected_dynamic_energy_nj_per_request"]*1e-6)
        assert math.isclose(r["fom_per_ns_mw"], 1/(r["average_latency_ns"]*r["average_power_mw"]))
        assert math.isclose(r["system_average_power_mw"], r["dynamic_power_mw"]+r["static_power_mw"]+r["refresh_power_mw"])
        assert r["refresh_power_mw"] == 0
        assert all(math.isfinite(r[k]) and r[k] > 0 for k in ("average_latency_ns", "average_power_mw", "dynamic_energy_pj_per_request"))
        if case not in CASES:
            assert r["hit_rates"] == suite["case_reports"]["optimized"]["hit_rates"]


def main():
    reports = {w: json.loads((ROOT/f"results/scope_v9_corrected_{w}.json").read_text())
               for w in ("attention", "ffn")}
    for report in reports.values(): validate(report)
    for workload, report in reports.items():
        old = ROOT/f"results/scope_v9_{workload}.json"
        archive = ROOT/f"results/scope_v9_before_access_audit_{workload}.json"
        if old.exists() and not archive.exists():
            archive.write_text(old.read_text())
        old.write_text(json.dumps(report, indent=2)+"\n")
    publish_v9()
    text = "# 访问口径核查与修正\n\n"
    text += "## 1. 三个问题的结论\n\n"
    text += (
        "**单次访问的功耗：你的 20 fJ / 1 ns = 20 μW 是正确的。** 这是脉冲期间的平均功率，不是可由这两个数推断的峰值。"
        "当前单元库的能量被定义为 fJ/bit/access，而缓存请求以 128 B 整行为单位。这个粒度假设必须与原始器件论文核对，不能把单 bit 的能量当作整行的能量。[^library]\n\n"
    )
    text += table(["仅单元写入示例", "并行写入位数", "总能量", "时长", "访问期间平均功率"],
                  [["单 bit",1,"20 fJ","1 ns","20 μW"],
                   ["16 B 数据",128,"2.56 pJ","1 ns","2,560 μW = 2.56 mW"],
                   ["128 B 缓存行",1024,"20.48 pJ","1 ns","20,480 μW = 20.48 mW"]])+"\n\n"
    text += (
        "这张示例表没有外围电路。实际 tag、译码、位线、SA、NoC 和片外访问都会贡献能量；即使 ISA 操作只用到一行中的一部分，行填充也可能搬运整行。"
        "没有把能量再乘以整个缓存容量，也没有把 mW 数字直接改写成 μW。\n\n"
        "**命中率：旧模型确实没有展开实际 tile 复用。** 已改为完整的 Q/K/V 和输出投影 GEMM、RoPE、因果 FlashAttention，以及 gate/up、SwiGLU、down。"
        "GEMM 使用可配置的 grouped CTA 顺序，group_m=8；A 为 [M,K]、Linear 权重为 [N,K]，K 方向逐块读入，累加留在寄存器/共享工作区，最后才写出结果。"
        "FlashAttention 不生成落在全局内存中的 S×S 分数矩阵。[^triton][^fa]\n\n"
        "已去掉事后添加的 0.88 跨帧 L3 复用概率。默认从冷缓存完整执行一次算子，而不是随机抽样一小段地址。"
        "这仍然是行为级地址序列，不是 OpenVLA 在真实 GPU 上采集的 trace。\n\n"
        "条件命中率的分母不同：$h_i=H_i/A_i$；$A_{i+1}=A_i-H_i$。例如 L1=90%、L2=50%、L3=20%，对应全局命中比例90%、5%、1%和片外4%，完全守恒。"
        "因此 **L3>L2>L1 不是必须满足的规律**。当 L2 吃掉多数复用请求时，L3 剩下的请求往往是首次读入的大权重，条件命中率反而可能低。\n\n"
        "规则矩阵在简单地址取模下可能撞到少数组。新增通用 XOR 折叠映射，默认使用它，取模方式仍可选。"
        "Accel-Sim 也提供线性/XOR/多项式映射选项；SCOPE 的实现只是可复现的通用假设，不声称等同于 Orin 的私有映射。"
        "SCOPE 仍采用三层 write-back/write-allocate、单 CTA 串行顺序，没有模拟真实 GPU 的 sector、并发 SM、L1 write-through 或 SMEM 分区。[^accel]\n\n"
        "**延迟：已统一平均值和单条路径。** 在 Sequential 查找方式下，miss 层只查 tag；命中层读取数据；store miss 向下取回数据后在 L1 提交。"
        "load 的后台回填不推迟响应，但仍计能量；脏逐出另计源数据读取、传输、目标写入的能量。"
        "之前把每个到达层都按完整命中操作计延迟、对下层套用同一读写比例、漏计回填能量，均已修正。[^destiny]\n\n"
        "另修复 SOT-MRAM 后端写能量仍为252 fJ/bit的问题：由当前单元库自动生成102 fJ/bit（0.102 pJ）和10 ns脉冲参数；不是手工修改最终结果。[^library]\n\n"
    )
    text += "## 2. 新结果与功耗窗口\n\n"
    text += (
        "响应延迟：$\\bar L=\\sum_{o,d}p(o,d)L_{o,d}$，其中 load/store 与 L1/L2/L3/OFF 的联合概率由实际重放计数得到。\n\n"
        "辅助诊断的单次访问动态功率：$P_{access}=\\sum E_{work}/\\sum T_{work}$。$T_{work}$ 包含该请求引起的读取、传输、回填、按计数分摊的脏逐出服务时间，"
        "假设这些工作串行执行。这是访问序列的时间加权平均功率；不是单个请求功率的算术平均，也不是峰值，不能把后台工作能量除以提前返回的 load 响应时间。"
        "它不使用每秒执行几次的频率，也不包含常驻泄漏/刷新。\n\n"
        "若将完全相同的时间窗同时定义为latency与功率分母，那么 $latency\\times power=energy$，FoM实质上退化为能效。"
        "这里响应延迟与包含后台工作的总服务时间不同，因此两者都列出，不把更低的访问功率自动解释为更省能量；能量列是独立的核查依据。\n\n"
        "默认恢复最初v9的系统平均功率 $P_{system}=\\lambda\\bar E+P_{static}+P_{refresh}$，FoM=$1/(\\bar L P_{system})$。"
        "活动量按用户选择恢复legacy_v9_analytical：5 Hz乘以原v9近似事务预算得到lambda，不乘完整重放事务数。Attention/FFN预算分别为1,917,056/2,373,136次每算子，权重按一次计。每种工作负载的四配置使用相同lambda。"
        "命中率与每请求能量继续取完整重放的统计，因此Power是旧活动预算场景估计，不是完整tile算子以5 Hz执行的实测或预测功耗。"
        "单次E/T不参与默认排名，避免将更长的等待或后台写入时间作为节能收益。下表使用mW，保留E/T诊断列以便核对。\n\n"
    )
    rows, phases, paths, clocks, cells, before = [], [], [], [], [], []
    for workload, suite in reports.items():
        original_path = ROOT/f"results/scope_v9_before_access_audit_{workload}.json"
        original = json.loads(original_path.read_text()) if original_path.exists() else None
        for case in CASES:
            r = suite["case_reports"][case]
            rows.append([workload, LABELS[case], *[f"{v*100:.2f}%" for v in r["hit_rates"]],
                         f"{r['offchip_reach_probability']*100:.2f}%", f"{r['average_latency_ns']:.4f}",
                         f"{r['dynamic_energy_pj_per_request']:.3f}", f"{r['serialized_service_time_ns_per_request']:.4f}",
                         f"{r['single_access_dynamic_power_mw']:.3f}", f"{r['fom_per_ns_mw']:.8g}",
                         f"{r['average_power_mw']:.3f}"])
            if original:
                old = original["case_reports"][case]
                before.append([workload,LABELS[case],f"{old['hit_rates'][1]*100:.2f}%",f"{r['hit_rates'][1]*100:.2f}%",
                               f"{old['average_latency_ns']:.4f}",f"{r['average_latency_ns']:.4f}"])
        r = suite["case_reports"]["optimized"]
        tr = r["hit_rate_model"]["trace_metadata"]
        clocks.append([workload,tr["sample_accesses"],tr["warmup_accesses"],f"{r['workload_access_mix']['rho_r']*100:.4f}%",
                       f"{tr['memory_access_rate_per_s']:.0f}",tr["indexing"],tr["group_m"]])
        for p in tr["phases"]:
            remaining = p["measured_requests"]
            rates = []
            for hits in p["hit_counts"][:3]:
                rates.append(f"{hits/remaining*100:.2f}%" if remaining else "N/A")
                remaining -= hits
            phases.append([workload,p["name"],p["loads"],p["stores"],*rates,p["hit_counts"][3]])
        for p in r["request_path_costs"]:
            paths.append([workload,p["op"],p["hit_level"],p["count"],f"{p['probability']*100:.5f}%",
                          f"{p['latency_ns']:.5f}",f"{p['dynamic_energy_nj']*1000:.3f}",
                          f"{p['serialized_service_time_ns']:.5f}",f"{p['access_dynamic_power_mw']*1000:.2f}"])
        for l in r["layers"]:
            # The finalized per-layer equations already include the selected circuits.
            eq = next(x for x in r["per_layer_access_equations"] if x["layer"] == l["name"])
            cells.append([workload,l["name"],l["device"],l["line_bytes"]*8,
                          f"{eq['write_energy_nj']*1000:.3f}", f"{eq['write_latency_ns']:.5f}",
                          f"{eq['write_energy_nj']/eq['write_latency_ns']*1e6:.2f}"])
    text += table(["算子","配置","L1条件命中","L2条件命中","L3条件命中","片外到达", "响应 ns", "每请求 pJ", "总服务 ns", "E/T诊断 mW", "FoM 1/(ns·mW)", "Power mW"], rows)+"\n\n"
    text += "5 ns 与 SRAM 10–20 ns 是设计目标，不是仿真输入，也没有强行将输出截断到这些区间。仍保留历史 L3 OSFET 0.15 延迟缩放和原有 NoC/SA 假设，结果不能作为硬件实测性能保证。\n\n"
    ffn = reports["ffn"]["case_reports"]["optimized"]
    lpddr_component = ffn["offchip_reach_probability"] * ffn["off_chip"]["latency_ns"]
    text += (f"S-A-O 的 FFN 仍为 {ffn['average_latency_ns']:.4f} ns，未达到5 ns：其中仅片外到达率×LPDDR延迟就占 {lpddr_component:.4f} ns。"
             "要继续压低延迟，需要减少实际片外请求或提供经过标定的新存储/互连时序，不能只调低显示值。\n\n")
    text += table(["算子","配置","旧 L2 条件命中","新 L2 条件命中","旧响应 ns","新响应 ns"],before)+"\n\n"
    text += "下表为 S-A-O 完整重放统计。读写比例来自发出的缓存事务，不是原始张量字节数之比，也不是固定3:1。\n\n"
    text += table(["算子","事务数","预热事务","读取比例","长期功耗所用事务/秒","映射","group_m"],clocks)+"\n\n"
    text += table(["算子","阶段","load事务","store事务","L1条件命中","L2条件命中","L3条件命中","片外事务"], phases)+"\n\n"
    text += "S-A-O 每条访问路径如下；路径行不包括依赖逐出状态的回写，它们已按实际计数加到上面总表的能量与总服务时间。\n\n"
    text += table(["算子","操作","数据来源","次数","占全部请求","响应 ns","动态 pJ","串行服务 ns","访问动态 μW"], paths)+"\n\n"
    text += "再看各层128 B整行写入的电路值，能量包括该层外围，不等于一个bit的写入值：\n\n"
    text += table(["算子","层","器件","行位数","写入 pJ","写入 ns","写入期间 μW"],cells)+"\n\n"
    text += "## 3. 验证与代码位置\n\n"
    text += (
        "验证包括：小尺寸FFN的每个地址与独立Python嵌套循环逐个核对；C++缓存结果与独立LRU重放核对；load/store路径、填充和片外请求计数守恒；"
        "20 fJ/1 ns和1024bit换算测试；改变调用频率时单次功率不变、长期动态功率同比例变化；回填是否在关键路径只改变响应延迟、不改变能量；"
        "四配置和三个开关重跑，逐项验证FoM与单位。此类验证证明实现符合模型，不能代替GPU实测或SPICE标定。\n\n"
        "```bash\nmake CXXFLAGS='-std=c++17 -Wall -O3' scope_model\npython3 -m unittest discover -s tests -v\n"
        "python3 scope.py config/scope_v9.json --compare --json-output results/scope_v9_corrected_attention.json\n"
        "python3 scope.py config/scope_v9.json --workload ffn --compare --json-output results/scope_v9_corrected_ffn.json\n"
        "python3 scripts/access_audit.py\n```\n\n"
    )
    for path, marker in (("model/TiledTrace.cpp","void AccessTrace::gemm"),
                         ("model/CacheModel.cpp","std::uint64_t CacheHierarchy::Cache::set_index"),
                         ("scope/core.py","    def _path_probabilities"),
                         ("scope/core.py","        dynamic_power_mw = access_rate"),
                         ("scope/power.py","def access_power_uw")):
        source = (ROOT/path).read_text().splitlines()
        index = next(i for i,l in enumerate(source) if marker in l)
        text += f"[{path}](<{ROOT/path}:{index+1}>)\n\n```{'cpp' if path.endswith('.cpp') else 'python'}\n"+"\n".join(source[index:index+7])+"\n```\n\n"
    text += (
        f"[^library]: 当前参数约定与SOT同步开关：[device_library_v9.json](<{ROOT/'config/device_library_v9.json'}>)；原始截图列为fJ，最终器件结论仍需原论文核验bit/宏单元粒度。\n"
        "[^triton]: [Triton 官方矩阵乘法教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)，提供分块循环与L2友好的grouped CTA调度。SCOPE未运行该GPU内核。\n"
        "[^fa]: [FlashAttention 原始论文](https://arxiv.org/abs/2205.14135)；[官方实现](https://github.com/Dao-AILab/flash-attention)。SCOPE仅重放相应数据搬运，不执行浮点运算。\n"
        "[^accel]: [Accel-Sim / GPGPU-Sim 4.x 设计说明](https://github.com/accel-sim/accel-sim-framework/blob/dev/gpu-simulator/gpgpu-sim4.md)，见缓存映射和写策略。\n"
        f"[^destiny]: 本地[component/Result.cpp](<{ROOT/'component/Result.cpp'}>)的Sequential分支：miss只访问tag，hit包含tag与数据阵列。\n"
    )
    (ROOT/"access-audit.md").write_text(text)
    print("Validated both workloads, four architectures and feature ablations; reports published locally.")


if __name__ == "__main__": main()
