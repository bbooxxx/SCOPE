"""Re-evaluate disclosed parameter scenarios without changing the v9 baseline."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scope.core import (DestinyRunner, build_evaluation_cases, comparison_summary,
                        evaluate_design, load_model_library, select_workload)
from scripts.v9_reports import relative_fom_gain, table


def scenarios():
    result = [("baseline", {})]
    for name, values in (
        ("asy_ion", [49.65, 66.2]), ("asy_write_ns", [1, 2]),
        ("asy_delta_v", [0.05]), ("osfet_ion", [3, 4.5, 9]),
        ("osfet_write_ns", [12, 15, 20, 30]),
        ("link_length_mm", [0.25, 0.5, 2]),
        ("sot_write_ns", [1, 2, 5]), ("sot_write_fj", [50]),
        ("l3_energy_scale", [0.75, 0.5, 0.25, 0.1]),
    ):
        result.extend((f"{name}={value:g}", {name: value}) for value in values)
    result.extend((f"joint_write={value}", {"asy_write_ns": 2,
                                           "osfet_write_ns": value})
                  for value in (12, 15, 20))
    result.extend((f"compact_l3={value:g}", {"asy_write_ns": 2,
                       "link_length_mm": 0.25, "l3_energy_scale": value})
                  for value in (0.5, 0.25, 0.1))
    return result


def apply_parameters(config, library, parameters):
    config, library = copy.deepcopy(config), copy.deepcopy(library)
    paths = {
        "asy_ion": ("AsyFET-eDRAM", "read_circuit", "ion_ua_per_um"),
        "asy_write_ns": ("AsyFET-eDRAM", "write_latency", "value_ns"),
        "asy_delta_v": ("AsyFET-eDRAM", "read_circuit", "delta_v"),
        "osfet_ion": ("OSFET-eDRAM", "read_circuit", "ion_ua_per_um"),
        "osfet_write_ns": ("OSFET-eDRAM", "write_latency", "value_ns"),
        "sot_write_ns": ("SOT-MRAM", "write_latency", "value_ns"),
        "sot_write_fj": ("SOT-MRAM", "write_energy", "value_fj_per_bit"),
    }
    for name, value in parameters.items():
        if not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            raise ValueError(f"invalid scenario value: {name}")
        if name == "l3_energy_scale":
            if value > 1:
                raise ValueError("energy scale must be at most one")
            layer = config["layers"][2]
            overrides = layer.setdefault("device_overrides", {})
            for device in library["devices"]:
                calibration = overrides.setdefault(device, {}).setdefault("circuit_calibration", {})
                calibration.update(read_energy_scale=value, write_energy_scale=value)
        elif name == "link_length_mm":
            for link in config["crossbars"]:
                link["link_length_mm"] = value
        elif name in paths:
            device, section, key = paths[name]
            library["devices"][device][section][key] = value
        else:
            raise ValueError(f"unknown scenario parameter: {name}")
    return config, library


def evaluate(config, library, runner):
    summaries, breakdown = [], {}
    for case in build_evaluation_cases(config):
        if case["group"] != "architecture":
            continue
        _, report = evaluate_design(
            case["config"], ROOT, runner, auto_build=False,
            device_library=library["devices"], model_library=library,
            library_path=ROOT / "config/device_library_v9.json", explore=True)
        summaries.append(comparison_summary(case["id"], report))
        breakdown[case["id"]] = {
            name: report[name] for name in
            ("latency_breakdown", "dynamic_power_breakdown", "static_power_breakdown",
             "per_layer_access_equations", "exploration")}
    by_case = {item["case"]: item for item in summaries}
    sao = by_case["optimized"]
    gains = {case: relative_fom_gain(sao["fom_per_ns_mw"], by_case[case]["fom_per_ns_mw"])
             for case in ("sram_osfet_osfet", "all_sram")}
    smm, sss = by_case["sram_mram_mram"], by_case["all_sram"]
    smm_gains = {"fom": relative_fom_gain(smm["fom_per_ns_mw"], sss["fom_per_ns_mw"]),
                "latency_reduction": 1 - smm["average_latency_ns"] / sss["average_latency_ns"]}
    return {"architectures": summaries, "gains": gains,
            "smm_gains": smm_gains,
            "meets_targets": all(item["feasible"] for item in summaries)
            and gains["sram_osfet_osfet"] >= 0.1 and gains["all_sram"] >= 0.8,
            "breakdown": breakdown}


def main():
    config_path = ROOT / "config/scope_v9.json"
    library_path = ROOT / "config/device_library_v9.json"
    base = json.loads(config_path.read_text())
    library = load_model_library(library_path, ROOT)
    runner = DestinyRunner(ROOT, ROOT / "destiny")
    results, rows, smm_rows = [], [], []
    for name, parameters in scenarios():
        changed, changed_library = apply_parameters(base, library, parameters)
        item = {"scenario": name, "parameters": parameters, "workloads": {}}
        for workload in ("attention", "ffn"):
            report = evaluate(select_workload(changed, workload), changed_library, runner)
            item["workloads"][workload] = report
            by_case = {x["case"]: x for x in report["architectures"]}
            sao, soo = by_case["optimized"], by_case["sram_osfet_osfet"]
            rows.append([name, workload,
                         f'{sao["average_latency_ns"]:.4f}', f'{sao["average_power_mw"]:.4f}',
                         f'{soo["average_latency_ns"]:.4f}', f'{soo["average_power_mw"]:.4f}',
                         f'{100*report["gains"]["sram_osfet_osfet"]:+.2f}%',
                         f'{100*report["gains"]["all_sram"]:+.2f}%',
                         "Yes" if report["meets_targets"] else "No"])
            smm, sss = by_case["sram_mram_mram"], by_case["all_sram"]
            floor = smm["offchip_reach_probability"] * changed["off_chip"]["latency_ns"]
            if name == "baseline" or name.startswith("sot_"):
                smm_rows.append([name, workload, f'{smm["average_latency_ns"]:.4f}',
                                 f'{smm["average_power_mw"]:.4f}',
                                 f'{100*report["smm_gains"]["fom"]:+.2f}%',
                                 f'{100*report["smm_gains"]["latency_reduction"]:+.2f}%',
                                 f'{floor:.4f}', f'{0.8*sss["average_latency_ns"]:.4f}'])
            print(name, workload, rows[-1][-3:], flush=True)
        results.append(item)
    output = {"baseline_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (config_path, library_path)},
              "semantics": "Hypothetical sensitivity scenarios, not measured device characterizations; all architectures share each scenario's device library and workload.",
              "results": results}
    (ROOT / "results/scope_v9_sensitivity.json").write_text(json.dumps(output, indent=2) + "\n")
    passing = [item["scenario"] for item in results
               if all(r["meets_targets"] for r in item["workloads"].values())]
    report = "# S-A-O 相对 S-O-O 的 FoM 敏感性分析\n\n"
    report += "FoM=1/(latency×power)，增幅=FoM_SAO/FoM_reference−1。目标是两种负载分别达到 S-O-O +10%、S-S-S +80%。\n\n"
    report += "保持 Orin 面积预算、128 B line、工作负载、命中率、OSFET 无刷新及既有 L3 0.15 延迟缩放不变。每个情景对四种架构使用同一份器件库；所有包含 OSFET 的层同时改变，不只修改 S-O-O。L2/L3 均重新选择阵列搜索目标和兼容 SA。\n\n"
    report += "这些扫描点是条件分析，不是新增实测数据，未覆盖 v9 默认参数。降低写入时间不自动降低写能量；提高 Ion 不自动更改密度，代表同面积器件改进要求，仍需验证。缩短连线只评估现有模型中的连线能耗，固定流水线延迟不随之缩短。\n\n"
    report += "两种负载均达标的情景：" + ("、".join(passing) or "无") + "。\n\n"
    report += "是否达标以本次扫描表为准。器件参数和能量缩放是待验证的假设，不作为实测结果或默认配置。\n\n"
    report += table(["参数情景", "负载", "SAO latency (ns)", "SAO power (mW)",
                     "SOO latency (ns)", "SOO power (mW)", "FoM vs SOO", "FoM vs SSS", "两项目标达标"], rows)
    report += "\n\n## S-M-M 对 S-S-S 的 20% 目标\n\n"
    report += "延迟降低率=1−L_SMM/L_SSS。下限仅保留片外延迟贡献，将所有片上操作按零计；若下限仍高于目标，仅改片上读写时间不能达标，必须减少片外访问。原密度为 140/100=1.4 倍。\n\n"
    report += "sot_* 为条件参数扫描。当前 v9 通过 sync_destiny_write_cell 将库中的写入能量与脉冲时间同步到生成的 DESTINY 单元文件，重新计算阵列并保留外围开销。它不代表器件改进已获实测验证。\n\n"
    report += table(["参数情景", "负载", "SMM latency (ns)", "SMM power (mW)",
                     "FoM vs SSS", "延迟降低 vs SSS", "乐观延迟下限 (ns)", "20% 延迟目标 (ns)"], smm_rows)
    report += "\n\n## 参数含义与依据\n\n"
    report += "- asy_ion / osfet_ion：单位 μA/μm；asy_write_ns / osfet_write_ns：单元写延迟；asy_delta_v：读摆幅，单位 V；link_length_mm：两条 NoC 连线长度。joint_write 同时将 AsyFET 写延迟设为 2 ns，并设置相应 OSFET 写延迟。\n"
    report += "- l3_energy_scale：所有架构的 L3 读写能量预算相对原值的比例，不改变延迟；compact_l3 同时将 AsyFET 写延迟设为 2 ns、两条连线设为 0.25 mm。该比例不是新测量值，0.25 表示需要降低 75% 读写能量的激进电路目标；必须证明能同时保持当前延迟和可靠性，不能把扫描达标视作器件已经实现。\n"
    report += "- [Zhang et al., 2023](https://onlinelibrary.wiley.com/doi/10.1002/aelm.202300150) 是原库 OSFET 6 μA/μm 的来源；其偏置与电容条件不能直接证明所有阵列都具有相同访问时间。本扫描的 3/4.5/9 μA/μm 和 12–30 ns 是敏感性假设，不冒充该文实测。\n"
    report += "- [MICRO 2001: Reducing Set-Associative Cache Energy via Way-Prediction and Selective Direct-Mapping](https://www.microarch.org/micro34/abstracts/powell.html) 说明串行 tag/data 与选择性激活存在能耗—延迟折中，可作为下一步电路优化方向；本次未把该论文的收益数值直接移植进 SCOPE。\n"
    report += "- [imec 2024](https://www.imec-int.com/en/articles/bringing-sot-mram-technology-closer-last-level-cache-memory-specifications) 报告单器件可达 300 ps 切换、低于 100 fJ/bit 的切换能量，并强调阵列集成仍需优化。sot_write_ns=1/2/5、sot_write_fj=50 是分别变化的探索点，不能据此声称同一宏同时达到这些数值；外围开销未清零，密度未更改。\n"
    report += "\n## 当前模型的限制\n\n平均响应和单条指令采用相同的load/store路径；回填和脏逐出的能量另行计入，默认功耗为相同工作负载频率下的动态、静态及刷新之和。trace是分块算子的串行地址重放，并非实测GPU trace；仍需验证并发CTA调度、缓存映射与器件标定。以上扫描不能代替这些校验。\n"
    report += "\n复现：`python3 scripts/v9_sensitivity.py`。原始结果及每组的延迟/功耗分解保存在 `results/scope_v9_sensitivity.json`。\n"
    (ROOT / "sensitivity.md").write_text(report)


if __name__ == "__main__":
    main()
