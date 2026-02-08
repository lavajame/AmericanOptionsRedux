import os
import sys
import time
import matplotlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from american_options.ju_1998 import Ju1998Pricing
from american_options.kim_jang_iter import KimJangIterative
from american_options.gauss_kronrod import GKRule
from american_options.american_put_gk import AmericanPutBoundaryGK
from american_options.boundary_estimator_piecewise import (
    gauss_newton_bspline,
    build_bspline_knots,
    build_bspline_basis_matrix,
    boundary_from_bspline,
    price_american_from_boundary,
)

GLOBAL_M=300
GLOBAL_N=600
GLOBAL_MAX_ITER=50


def _output_path(filename):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


def _make_time_grid(T, n_steps, cluster_power=2.5):
    u = np.linspace(0.0, 1.0, n_steps + 1)
    times = T * (1.0 - (1.0 - u) ** cluster_power)
    times[0] = 0.0
    times[-1] = T
    return times

BENCHMARK_FDM_PRICES = [
    0.2188175852,
    1.3845248237,
    4.77996625,
    11.0960855904,
    20.0,
    10.2337739433,
    7.0321566464,
    0.0062129953,
    0.2001025201,
    0.4323213514,
    0.851945671,
    1.5792062481,
    1.9897014319,
]

BENCHMARK_FDM_BOUNDARIES = {
    "put": {
        "times": [
            0.0005, 0.0105, 0.0205, 0.031, 0.041, 0.051, 0.0615, 0.0715,
            0.082, 0.092, 0.102, 0.1125, 0.1225, 0.133, 0.143, 0.153, 0.1635,
            0.1735, 0.1835, 0.194, 0.204, 0.2145, 0.2245, 0.2345, 0.245, 0.255,
            0.2655, 0.2755, 0.2855, 0.296, 0.306, 0.3165, 0.3265, 0.3365, 0.347,
            0.357, 0.367, 0.3775, 0.3875, 0.398, 0.408, 0.418, 0.4285, 0.4385,
            0.449, 0.459, 0.469, 0.4795, 0.4895, 0.5,
        ],
        "values": [
            97.90509433, 93.77632793, 91.78698008, 90.30094699, 89.15193636,
            88.21974579, 87.34439019, 86.60889531, 85.90847837, 85.30802346,
            84.75397424, 84.22845922, 83.74888706, 83.25969797, 82.84959232,
            82.45333412, 82.02614787, 81.67026656, 81.33723557, 81.00325591,
            80.67300422, 80.33623099, 80.04123555, 79.76274491, 79.47879272,
            79.21529274, 78.93332584, 78.66442164, 78.40919252, 78.16199853,
            77.9597825, 77.7293432, 77.48288665, 77.26756112, 77.09090021,
            76.85989885, 76.64813552, 76.48851235, 76.26796474, 76.06599341,
            75.92350403, 75.70709054, 75.55409596, 75.36697511, 75.18837016,
            75.04331726, 74.86000093, 74.72985453, 74.53914464, 74.42320083,
        ],
    },
    "call": {
        "times": [
            0.0005, 0.0105, 0.0205, 0.031, 0.041, 0.051, 0.0615, 0.0715,
            0.082, 0.092, 0.102, 0.1125, 0.1225, 0.133, 0.143, 0.153, 0.1635,
            0.1735, 0.1835, 0.194, 0.204, 0.2145, 0.2245, 0.2345, 0.245, 0.255,
            0.2655, 0.2755, 0.2855, 0.296, 0.306, 0.3165, 0.3265, 0.3365, 0.347,
            0.357, 0.367, 0.3775, 0.3875, 0.398, 0.408, 0.418, 0.4285, 0.4385,
            0.449, 0.459, 0.469, 0.4795, 0.4895, 0.5,
        ],
        "values": [
            102.17243555, 106.65669256, 108.9850807, 110.75941479, 112.17896937,
            113.38483803, 114.48461278, 115.46984047, 116.36469399, 117.17757677,
            117.9541295, 118.72390055, 119.42211856, 120.07866203, 120.70587031,
            121.28923814, 121.89397958, 122.40194577, 122.97051873, 123.49576349,
            123.96159018, 124.474588, 124.9528925, 125.40595082, 125.8606398,
            126.28144672, 126.70842994, 127.11128796, 127.50727138, 127.9108943,
            128.29357352, 128.68373416, 129.05128029, 129.40853189, 129.77200291,
            130.10235206, 130.41213974, 130.76129888, 131.10230598, 131.44773581,
            131.76031026, 132.04343354, 132.35556863, 132.67617891, 132.99164614,
            133.25909163, 133.53692142, 133.85455786, 134.12925076, 134.38555775,
        ],
    },
}


def price_bspline_eeb(S0, K, r, q, sigma, T, option_type):
    w = 1 if option_type == "call" else -1
    N_time = 160
    times = _make_time_grid(T, N_time, cluster_power=2.5)
    degree = 3
    n_basis = 10
    knots = build_bspline_knots(T, n_basis, degree, knot_strategy="power", power=2.5)
    tau = T - times
    basis_matrix = build_bspline_basis_matrix(tau, knots, degree, add_sqrt=True)

    coeffs0 = np.full(n_basis + 1, K)
    coeffs0[-1] = 0.0
    n_residuals = len(coeffs0) + 4

    coeffs_opt = gauss_newton_bspline(
        coeffs0, basis_matrix, times, K, r, q, sigma, w,
        n_int_steps_eep=60, n_residuals=n_residuals,
        lam=1e-3, tol=1e-6, max_iter=12
    )

    B = boundary_from_bspline(times, T, coeffs_opt, knots, degree, basis_matrix=basis_matrix)
    B[-1] = K

    return price_american_from_boundary(
        S0, K, r, q, sigma, T, times, B, w, n_int_steps_eep=80
    )


def boundary_bspline_eeb(S0, K, r, q, sigma, T, option_type):
    w = 1 if option_type == "call" else -1
    N_time = 140
    times = _make_time_grid(T, N_time, cluster_power=2.5)
    degree = 4
    n_basis = 21
    knots = build_bspline_knots(T, n_basis, degree, knot_strategy="power", power=2.5)
    tau = T - times
    basis_matrix = build_bspline_basis_matrix(tau, knots, degree, add_sqrt=True)

    coeffs0 = np.full(n_basis + 1, K)
    coeffs0[-1] = 0.0
    n_residuals = len(coeffs0) + 2

    coeffs_opt = gauss_newton_bspline(
        coeffs0, basis_matrix, times, K, r, q, sigma, w,
        n_int_steps_eep=60, n_residuals=n_residuals,
        lam=1e-3, tol=1e-6, max_iter=12
    )

    B = boundary_from_bspline(times, T, coeffs_opt, knots, degree, basis_matrix=basis_matrix)
    B[-1] = K
    tau = T - times
    return tau, B

def run_price_comparison():
    # Comparison cases (from Ju 1999 Exhibit 4 and 3)
    cases = [
        # Calls: K=100, T=0.5
        {"S": 80, "K": 100, "r": 0.03, "q": 0.07, "T": 0.5, "sigma": 0.2, "type": "call"},
        {"S": 90, "K": 100, "r": 0.03, "q": 0.07, "T": 0.5, "sigma": 0.2, "type": "call"},
        {"S": 100, "K": 100, "r": 0.03, "q": 0.07, "T": 0.5, "sigma": 0.2, "type": "call"},
        {"S": 110, "K": 100, "r": 0.03, "q": 0.07, "T": 0.5, "sigma": 0.2, "type": "call"},
        {"S": 120, "K": 100, "r": 0.03, "q": 0.07, "T": 0.5, "sigma": 0.2, "type": "call"},
        {"S": 100, "K": 100, "r": 0.03, "q": 0.07, "T": 0.5, "sigma": 0.4, "type": "call"},
        {"S": 100, "K": 100, "r": 0.00, "q": 0.07, "T": 0.5, "sigma": 0.3, "type": "call"},
        # Puts: S=40, r=0.0488, q=0
        {"S": 40, "K": 35, "r": 0.0488, "q": 0.0, "T": 0.0833, "sigma": 0.2, "type": "put"},
        {"S": 40, "K": 35, "r": 0.0488, "q": 0.0, "T": 0.3333, "sigma": 0.2, "type": "put"},
        {"S": 40, "K": 35, "r": 0.0488, "q": 0.0, "T": 0.5833, "sigma": 0.2, "type": "put"},
        {"S": 40, "K": 40, "r": 0.0488, "q": 0.0, "T": 0.0833, "sigma": 0.2, "type": "put"},
        {"S": 40, "K": 40, "r": 0.0488, "q": 0.0, "T": 0.3333, "sigma": 0.2, "type": "put"},
        {"S": 40, "K": 40, "r": 0.0488, "q": 0.0, "T": 0.5833, "sigma": 0.2, "type": "put"},
    ]

    rows = []
    for idx, c in enumerate(cases):
        print(f"Processing case: {c}")
        t0 = time.perf_counter()
        ju = Ju1998Pricing(c["S"], c["K"], c["r"], c["T"], c["sigma"], c["q"], c["type"]).price()
        ju_time = time.perf_counter() - t0

        fdm = BENCHMARK_FDM_PRICES[idx]
        fdm_time = 0.0

        t0 = time.perf_counter()
        kim = KimJangIterative(
            K=c["K"],
            r=c["r"],
            T=c["T"],
            sigma=c["sigma"],
            alpha=c["q"],
            option_type=c["type"],
            gk_rule=GKRule.BALANCED,
        ).price(c["S"], c["T"])
        kim_time = time.perf_counter() - t0

        gk_time = np.nan
        gk_price = np.nan
        gk_boundary_T = np.nan
        t0 = time.perf_counter()
        gk_solver = AmericanPutBoundaryGK(
            K=c["K"],
            r=c["r"],
            q=c["q"],
            sigma=c["sigma"],
            T=c["T"],
            gk_rule=GKRule.BALANCED,
        )
        if c["type"] == "put":
            gk_tau, gk_boundary = gk_solver.compute_boundary(n_nodes=160, n_iter=10)
            gk_price = gk_solver.price_put(c["S"], c["T"], boundary=(gk_tau, gk_boundary))
            gk_boundary_T = float(gk_boundary[-1])
        else:
            gk_price = gk_solver.price_call(c["S"], c["T"])
            gk_tau, gk_boundary = gk_solver.compute_call_boundary(n_nodes=160, n_iter=10)
            gk_boundary_T = float(gk_boundary[-1])
        gk_time = time.perf_counter() - t0

        spline_time = np.nan
        spline_price = np.nan
        t0 = time.perf_counter()
        spline_price = price_bspline_eeb(
            c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"], c["type"]
        )
        spline_time = time.perf_counter() - t0

        rows.append({
            "Type": c["type"],
            "S": c["S"],
            "K": c["K"],
            "T": c["T"],
            "sigma": c["sigma"],
            "r": c["r"],
            "q": c["q"],
            "FDM": fdm,
            "Ju1998": ju,
            "KimJang": kim,
            "GKPrice": gk_price,
            "SplineEEB": spline_price,
            # "GK Boundary @T": gk_boundary_T,
            # "Diff ju - fdm": ju - fdm,
            # "Diff kim - fdm": kim - fdm,
            # "Diff gk - fdm": gk_price - fdm,
            # "Diff spline - fdm": spline_price - fdm,
            "t_ju_s": ju_time,
            # "t_fdm_s": fdm_time,
            "t_kim_s": kim_time,
            "t_gk_s": gk_time,
            "t_spline_s": spline_time,
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, na_rep="N/A", formatters={
        "Ju1998": "{:.10f}".format,
        "FDM": "{:.10f}".format,
        "KimJang": "{:.10f}".format,
        "GKPrice": (lambda x: f"{x:.10f}" if np.isfinite(x) else "N/A"),
        "SplineEEB": (lambda x: f"{x:.10f}" if np.isfinite(x) else "N/A"),
        # "GK Boundary @T": (lambda x: f"{x:.10f}" if np.isfinite(x) else "N/A"),
        # "Diff ju - fdm": "{:.10e}".format,
        # "Diff kim - fdm": "{:.10e}".format,
        # "Diff spline - fdm": (lambda x: f"{x:.10e}" if np.isfinite(x) else "N/A"),
        "t_ju_s": "{:.6f}".format,
        # "t_fdm_s": "{:.6f}".format,
        "t_kim_s": "{:.6f}".format,
        "t_gk_s": (lambda x: f"{x:.6f}" if np.isfinite(x) else "N/A"),
        "t_spline_s": (lambda x: f"{x:.6f}" if np.isfinite(x) else "N/A"),
    }))


def plot_boundary_comparison():
    cases = [
        {
            "label": "Put",
            "params": {
                "S": 100,
                "K": 100,
                "r": 0.035,
                "q": 0.035,
                "T": 0.5,
                "sigma": 0.2,
                "type": "put",
            },
        },
        {
            "label": "Call",
            "params": {
                "S": 100,
                "K": 100,
                "r": 0.035,
                "q": 0.035,
                "T": 0.5,
                "sigma": 0.2,
                "type": "call",
            },
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, case in zip(axes, cases):
        print(f"Processing case: {case['label']}")
        params = case["params"]

        bench = BENCHMARK_FDM_BOUNDARIES[params["type"]]
        times = bench["times"]
        values = bench["values"]

        ju = Ju1998Pricing(
            params["S"], params["K"], params["r"], params["T"], params["sigma"], params["q"], params["type"]
        )
        S_star = ju.solve_critical_price()


        gk_boundary = None
        gk_solver = AmericanPutBoundaryGK(
            K=params["K"],
            r=params["r"],
            q=params["q"],
            sigma=params["sigma"],
            T=params["T"],
            gk_rule=GKRule.BALANCED,
        )
        n_nodes = 200
        if params["type"] == "put":
            gk_tau, gk_boundary = gk_solver.compute_boundary(n_nodes=n_nodes, n_iter=10)
        else:
            gk_tau, gk_boundary = gk_solver.compute_call_boundary(n_nodes=n_nodes, n_iter=10)

        ju_boundary = [S_star for _ in times]

        # Smooth boundary for display by linear interpolation over time
        if len(times) > 1:
            t_dense = np.linspace(min(times), max(times), 300)
            b_dense = np.interp(t_dense, times, values)
            ax.plot(t_dense, b_dense, label="FDM EEB")
        else:
            ax.plot(times, values, label="FDM EEB")
        ax.plot(times, ju_boundary, "--", label="Ju1998 EEB (constant)")
        if gk_boundary is not None:
            ax.plot(gk_tau, gk_boundary, "-.", label="GK EEB (iterative)")

        spline_tau, spline_boundary = boundary_bspline_eeb(
            params["S"], params["K"], params["r"], params["q"],
            params["sigma"], params["T"], params["type"]
        )
        ax.plot(spline_tau, spline_boundary, ":", label="Spline EEB")
        ax.invert_xaxis()
        ax.set_xlabel("Time to maturity (years)")
        ax.set_ylabel("Early exercise boundary S*")
        ax.set_title(f"EEB Comparison ({case['label']})")
        ax.legend()

    out_path = _output_path("eeb_comparison_2.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot: {out_path}")


def plot_call_boundary_comparison():
    params = {
        "S": 100,
        "K": 100,
        "r": 0.03,
        "q": 0.07,
        "T": 0.5,
        "sigma": 0.2,
        "type": "call",
    }

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

    bench = BENCHMARK_FDM_BOUNDARIES["call"]
    times = bench["times"]
    values = bench["values"]

    ju = Ju1998Pricing(
        params["S"], params["K"], params["r"], params["T"], params["sigma"], params["q"], params["type"]
    )
    S_star = ju.solve_critical_price()

    gk_solver = AmericanPutBoundaryGK(
        K=params["K"],
        r=params["r"],
        q=params["q"],
        sigma=params["sigma"],
        T=params["T"],
        gk_rule=GKRule.BALANCED,
    )
    gk_tau, gk_boundary = gk_solver.compute_call_boundary(n_nodes=220, n_iter=10)

    ju_boundary = [S_star for _ in times]

    if len(times) > 1:
        t_dense = np.linspace(min(times), max(times), 300)
        b_dense = np.interp(t_dense, times, values)
        ax.plot(t_dense, b_dense, label="FDM EEB")
    else:
        ax.plot(times, values, label="FDM EEB")

    ax.plot(times, ju_boundary, "--", label="Ju1998 EEB (constant)")
    ax.plot(gk_tau, gk_boundary, "-.", label="GK EEB (duality boundary)")

    ax.invert_xaxis()
    ax.set_xlabel("Time to maturity (years)")
    ax.set_ylabel("Early exercise boundary S*")
    ax.set_title("Call EEB Comparison")
    ax.legend()

    out_path = _output_path("call_eeb_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    run_price_comparison()
    plot_boundary_comparison()
    # plot_call_boundary_comparison()
