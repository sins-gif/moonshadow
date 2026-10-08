"""附录数值的独立复核：从输入重算，与文档字面值逐格比对。

    python tools/audit_appendix.py

存在理由：附录 A.4 初版按手算写入时有 7 个数字在第 5–6 位有误，靠断言才发现。
本工具把「文档数字 ↔ 输入重算」的一致性变成一条可随时执行的命令，
并同时核对 A.3 的反例算式。

两处复核对象：
- A.3：加性方案的可分离性反例（定理 5）
- A.4：有界乘性方案的数值实例（定理 3）
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.scoring import RECENT_WINDOW, TIME_FLOOR  # noqa: E402

#: 文档给到 6 位小数，容差取半个末位。
TOL = 5e-7

failures: list[str] = []


def check(label: str, actual: float, documented: float) -> None:
    ok = abs(actual - documented) <= TOL
    if not ok:
        failures.append(f"{label}: 重算 {actual:.6f} ≠ 文档 {documented:.6f}")
    print(f"    {'✓' if ok else '✗'} {label:<26} 重算 {actual:.6f}   文档 {documented:.6f}")


def field_days(tier_days: float) -> float:
    return tier_days


def derive(i: float, s: float, mu: float, delta_days: float, h: float, f: float) -> dict[str, float]:
    sigma = (1 + s) / 2
    w_sigma = 0.60 * sigma
    w_i = 0.27 * (i / 10)
    w_mu = 0.13 * mu
    r = w_sigma + w_i + w_mu
    delta = 0.5 ** (delta_days / h)
    m = f + (1 - f) * delta
    return {
        "sigma": sigma, "w_sigma": w_sigma, "w_i": w_i, "w_mu": w_mu,
        "r": r, "delta": delta, "m": m, "q": r * m,
    }


def audit_a3() -> None:
    """A.3：加性方案的反例（定义 4）。

    两条候选都必须**窗口合法**——旧见证的 `P = (T5, Δ=120)` 超窗，窗口本身就会挡掉它，
    用它证明「θ 不可行」无效。现行见证取自用例集（`N` 来自 09，`P` 来自 17）。
    """
    print("A.3 反例（加性方案，定义 4）")
    negative = dict(i=2, s=0.15, mu=0.0, delta_days=10.0, h=90.0)
    positive = dict(i=3, s=0.72, mu=0.0, delta_days=30.0, h=14.0)

    def additive(c: dict) -> float:
        sigma = (1 + c["s"]) / 2
        delta = 0.5 ** (c["delta_days"] / c["h"])
        return 0.45 * sigma + 0.25 * delta + 0.20 * (c["i"] / 10) + 0.10 * c["mu"]

    a_n, a_p = additive(negative), additive(positive)
    check("N 的加性分数", a_n, 0.530219)
    check("P 的加性分数", a_p, 0.503608)
    if not a_n > a_p:
        failures.append("定理 5 的反例不成立：负例分数未高于正例")
    if positive["delta_days"] > RECENT_WINDOW:
        failures.append("定理 5 的正例超窗——它不是 θ 的证据，反例无效")
    print(f"    {'✓' if a_n > a_p else '✗'} 可分离区间为空              差值 {a_n - a_p:+.6f}")
    print(f"    召回 P 要求 θ < {a_p:.6f}；排除 N 要求 θ > {a_n:.6f} —— 互斥\n")


def audit_a4() -> None:
    """A.4：有界乘性方案的数值实例（方案 B 的输入：i = 4 与 5）。

    **两列 floor**：`TIME_FLOOR`（现行采纳值，文档正文用）与 `0.62`
    （`experiments/user_a4_candidate_a.py` / `_b.py` 硬编码的值，那两个脚本不得修改，
    因此这一列作为历史列保留并与脚本逐格对齐）。`σ`/`w_*`/`r`/`δ` 与 floor 无关。

    注意区分两个都叫「余量」的量：**比值余量** `m_B/m_A − r_A/r_B`（判定式两侧之差）
    与**分数余量** `q_B − q_A`。两者数值不同，引用时必须写明是哪一个。
    """
    print(f"A.4 数值实例（有界乘性；现行 f = {TIME_FLOOR}，附历史 f = 0.62）")
    inputs = {
        "A": dict(i=4, s=0.95, mu=0.5, delta_days=55.0, h=30.0,
                  sigma=0.975000, w_sigma=0.585000, w_i=0.108000, w_mu=0.065000,
                  r=0.758000, delta=0.280616),
        "B": dict(i=5, s=0.60, mu=0.5, delta_days=1.5, h=14.0,
                  sigma=0.800000, w_sigma=0.480000, w_i=0.135000, w_mu=0.065000,
                  r=0.680000, delta=0.928425),
    }
    #: floor → (每候选的 m/q, m_B/m_A, 比值余量, 分数余量)
    documented = {
        TIME_FLOOR: (
            {"A": (0.706851, 0.535793), "B": (0.970833, 0.660167)},
            1.373463, 0.258757, 0.124374,
        ),
        0.62: (
            {"A": (0.726634, 0.550788), "B": (0.972801, 0.661505)},
            1.338778, 0.224072, 0.110717,
        ),
    }

    for floor, (expected, exp_ratio_m, exp_ratio_margin, exp_score_margin) in documented.items():
        tag = "现行" if floor == TIME_FLOOR else "历史"
        print(f"  ── f = {floor}（{tag}）")
        computed = {}
        for name, doc in inputs.items():
            got = derive(doc["i"], doc["s"], doc["mu"], doc["delta_days"], doc["h"], floor)
            computed[name] = got
            print(f"    候选 {name}（i={doc['i']}）")
            for key in ("sigma", "w_sigma", "w_i", "w_mu", "r", "delta"):
                check(key, got[key], doc[key])
            check("m", got["m"], expected[name][0])
            check("q", got["q"], expected[name][1])

        ratio_m = computed["B"]["m"] / computed["A"]["m"]
        ratio_r = computed["A"]["r"] / computed["B"]["r"]
        score_margin = computed["B"]["q"] - computed["A"]["q"]
        print("    比值")
        check("m_B/m_A", ratio_m, exp_ratio_m)
        check("r_A/r_B", ratio_r, 1.114706)
        check("比值余量 m_B/m_A − r_A/r_B", ratio_m - ratio_r, exp_ratio_margin)
        check("分数余量 q_B − q_A", score_margin, exp_score_margin)
        if not ratio_m > ratio_r:
            failures.append(f"A.4（f={floor}）的判定不成立：B 未胜出")

    # 临界 floor 与 f 无关：m = δ + (1−δ) f，解 m_B/m_A = r_A/r_B。
    computed = {n: derive(d["i"], d["s"], d["mu"], d["delta_days"], d["h"], TIME_FLOOR)
                for n, d in inputs.items()}
    ratio_r = computed["A"]["r"] / computed["B"]["r"]
    d_a, d_b = computed["A"]["delta"], computed["B"]["delta"]
    f_star = (d_b - ratio_r * d_a) / (ratio_r * (1 - d_a) - (1 - d_b))
    print("  临界 floor（与 f 无关）")
    check("临界 floor f*", f_star, 0.842939)
    print(f"    {'✓' if f_star > 0.70 else '✗'} f* 高于 f=0.70 与 f={TIME_FLOOR}"
          f"    ⇒ 两个取值下都不反转\n")


def main() -> int:
    print(f"附录数值复核（容差 {TOL:g}）\n" + "=" * 74)
    audit_a3()
    audit_a4()
    print("=" * 74)
    if failures:
        print(f"复核失败：{len(failures)} 项")
        for item in failures:
            print(f"  ✗ {item}")
        return 1
    print("复核通过：全部字面值均可由输入重算得到")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
