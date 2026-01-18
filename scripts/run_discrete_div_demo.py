import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from american_options.american_put_gk import plot_discrete_div_boundary_comparison


def main() -> None:
    S0 = 100.0
    K = 100.0
    r = 0.02
    q = 0.01
    sigma = 0.2
    T = 1.0

    discrete_dividends = [
        (0.25, 10.0),
        # (0.50, 1.0),
        # (0.75, 1.0),
    ]

    # plot_discrete_div_boundary_comparison(
    #     S0=S0,
    #     K=K,
    #     r=r,
    #     q=q,
    #     sigma=sigma,
    #     T=T,
    #     option_type="put",
    #     discrete_dividends=discrete_dividends,
    #     n_nodes=80,
    #     n_iter=8,
    #     use_duality_for_call=True,
    #     fd_M=400,
    #     fd_N=2000,
    # )

    plot_discrete_div_boundary_comparison(
        S0=S0,
        K=K,
        r=r,
        q=q,
        sigma=sigma,
        T=T,
        option_type="call",
        discrete_dividends=discrete_dividends,
        n_nodes=80,
        n_iter=8,
        use_duality_for_call=True,
        fd_M=400,
        fd_N=2000,
    )


if __name__ == "__main__":
    main()
