import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
sns.set(rc={"figure.dpi": 150, "savefig.dpi": 300})
sns.set_context("notebook")
sns.set_style("ticks")
sns.set(
    rc={
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "font.size": 12,
        "axes.labelsize": 16,
        "axes.grid": False,
        "legend.fontsize": 10,
        "ytick.left": True,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "pgf.texsystem": "lualatex",
        "text.latex.preamble": r"\usepackage{xcolor}",
        "text.usetex": True,
    },
    style="whitegrid",
)


def iso_perf_curve(x1, y, params):
    """
    For a given value of the performance threshold `y` and
    the amount of translated data `x1` determines the 
    amount of manual data `x2` that leads to desired performance `y`

    Inputs:
        - x1 (int) : Amount of translated data
        - y (float) : Performance Value
        - params (list): A list containing 5 elements each representing a parameter of the AMUE performance function
    
    """

    a0 = params[0]
    a1 = params[1]
    alpha1 = params[2]
    a2 = params[3]
    alpha2 = params[4]
    return ((1 / a2) * (y - a0 - a1 * (x1 ** alpha1))) ** (1 / alpha2)


def iso_cost_curve(x1, c12, cprime):
    """
    Returns x2 = cprime - c12 * x1

    Inputs:
        - x1 (int) : Amount of translated data
        - c12 (float) : The ratio of unit costs of translated data and manual data
        - cprime (float) : The ratio of total allowed cost and the unit cost of manual data i.e. C / c2 
    """

    return cprime - c12 * x1


def get_min_cost_point_on_isoperf_curve(
    params, y, c12, x1_bounds=(0, 3696), step_size=1
):
    """
    Searches for the minimum cost point along the isoperf surface

    Inputs:
        - params (list) : A list containing 5 elements each representing a parameter of the AMUE performance function
        - y (float) : Performance Value
        - c12 (float) : The ratio of unit costs of translated data and manual data
    
    Returns (x1,x2) representing the lowest cost configuration of translated and manual data
    """

    x1s = np.linspace(
        x1_bounds[0], x1_bounds[1], (x1_bounds[1] - x1_bounds[0]) // step_size
    )
    min_cost = None
    mcp = None
    for x1 in x1s:
        x2 = iso_perf_curve(x1, y, params)
        cost = c12 * x1 + x2
        if min_cost is None or cost < min_cost:
            min_cost = cost
            mcp = (x1, x2)

    return mcp


def get_operating_point(params, y, c12, x1_max, max_iters=1000, threshold=1e-3):
    """
    Given the AMUE performance function's parameters,
    a desired performance threshold `y`,
    and the cost ratio between the two sources of data compute
    the minimum cost point
    
    Inputs:
        - params (list) : A list containing 5 elements each representing a parameter of the AMUE performance function
        - y (float) : Performance threshold at which min cost point is to be determined
        - c12 (float) : The ratio of unit costs of translated data and manual data
        - x1_max (int) : Maximum amount of translated data that is allowed. It will often be equal to the pivot size
        - max_iters (int) : Maximum number of iterations to search for operating point
        - threshold (float) : Convergence crieteria for determining the operating point

    Returns (x1,x2) representing the lowest cost configuration of translated and manual data
    """
    a0 = params[0]
    a1 = params[1]
    alpha1 = params[2]
    a2 = params[3]
    alpha2 = params[4]

    x1 = 0
    # import pdb

    # pdb.set_trace()
    for i in range(max_iters):
        x1_updated = (alpha1 * a1 / (alpha2 * a2 * c12)) ** (1 / (1 - alpha1)) * (
            (1 / a2) * (y - a0 - a1 * (x1 ** alpha1))
        ) ** ((1 - alpha2) / (alpha2 * (1 - alpha1)))

        if abs(x1_updated - x1) < threshold:
            break
        x1 = x1_updated
    x1_op = x1
    x2_op = iso_perf_curve(x1_op, y, params)

    # If the operating point crosses realizable region, search within realizable region
    if x1_op >= x1_max or np.isnan(x1_op) or np.isnan(x2_op):
        x1_op, x2_op = get_min_cost_point_on_isoperf_curve(
            params, y, c12, x1_bounds=(0, x1_max)
        )

    return x1_op, x2_op


def get_expansion_path(
    params,
    ys,
    c12,
    lang,
    x1_max,
    c1=0.0067,
    max_iters=1000,
    threshold=1e-3,
    step_size=5,
    plot=True,
    save_dir="outputs/exp_paths",
):
    """
    Computes the expansion path for a given performance function

    Inputs:
        - params (list) : A list containing 5 elements each representing a parameter of the AMUE performance function
        - ys (list) : A set of performance thresholds on which the least cost operating points are to be determined
        - c12 (float) : The ratio of unit costs of translated data and manual data
        - lang (str) : Language for which the expansion path is to be computed
        - x1_max (int) : Maximum amount of translated data that is allowed. It will often be equal to the pivot size
        - c1 (float) : Unit cost of translated data
        - max_iters (int) : Maximum number of iterations to search for operating point
        - threshold (float) : Convergence crieteria for determining the operating point
        - plot (bool) : Whether to plot the expansion path

    """
    op_points = []
    iso_perf_points = []
    iso_cost_points = []
    total_costs = []

    for y in ys:
        x1_op, x2_op = get_operating_point(
            params, y, c12, x1_max, max_iters=max_iters, threshold=threshold
        )

        cprime = x2_op + x1_op * c12
        x1 = np.linspace(0, x1_max * 10, (x1_max * 10) // step_size + 1)
        x2_iso_cost = iso_cost_curve(x1, c12, cprime)
        x2_iso_perf = iso_perf_curve(x1, y, params)
        op_points.append([x1_op, x2_op])
        total_costs.append(c1 * x1_op + c1 / c12 * x2_op)

        iso_perf_points.append(
            np.concatenate(
                [
                    x1[:, np.newaxis],
                    x2_iso_perf[:, np.newaxis],
                    y * np.ones([len(x1), 1]),
                ],
                axis=1,
            )
        )
        iso_cost_points.append(
            np.concatenate(
                [
                    x1[:, np.newaxis],
                    x2_iso_cost[:, np.newaxis],
                    cprime * c12 * np.ones([len(x1), 1]),
                ],
                axis=1,
            )
        )

    op_points = np.array(op_points)
    iso_perf_points_df = pd.DataFrame(np.concatenate(iso_perf_points, axis=0))
    iso_perf_points_df.columns = [
        "Translation Data Size",
        "Labelled Data Size",
        "F1-Score",
    ]

    iso_cost_points_df = pd.DataFrame(np.concatenate(iso_cost_points, axis=0))
    iso_cost_points_df.columns = ["Translation Data Size", "Labelled Data Size", "Cost"]

    if plot:
        plt.figure(figsize=(15, 10))
        sns.set(font_scale=3)
        sns.set_style("whitegrid")
        ax1 = sns.lineplot(
            x="Translation Data Size",
            y="Labelled Data Size",
            hue="F1-Score",
            data=iso_perf_points_df,
            palette="Reds",
            legend=False,
        )

        ax2 = sns.lineplot(
            x="Translation Data Size",
            y="Labelled Data Size",
            hue="Cost",
            palette="crest",
            data=iso_cost_points_df,
            legend=False,
        )
        norm = plt.Normalize(
            iso_perf_points_df["F1-Score"].min(), iso_perf_points_df["F1-Score"].max()
        )
        sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
        sm.set_array([])

        norm = plt.Normalize(
            iso_cost_points_df["Cost"].min(), iso_cost_points_df["Cost"].max()
        )
        sm2 = plt.cm.ScalarMappable(cmap="crest", norm=norm)
        sm2.set_array([])

        cbar1 = ax1.figure.colorbar(sm)
        cbar1.ax.set_ylabel(
            "Performance $\Pi$ along Isoperfs", rotation=270, labelpad=30
        )
        cbar2 = ax1.figure.colorbar(sm2, orientation="horizontal", pad=0.2)
        cbar2.ax.set_xlabel("Net Cost of Operation $C$ along Isocosts")

        plt.plot(
            op_points[:, 0],
            op_points[:, 1],
            marker="o",
            color="black",
            label="Expansion Path",
            linewidth=2.5,
        )
        plt.plot(x1, x1, label="$M = T$", color="darkgreen", linewidth=2.5)
        plt.ylim(0, op_points[:, 1].max())
        plt.xlim(0, (x1_max * 10))

        plt.annotate(
            "$P$",
            xy=(x1_max, 0),
            xycoords="data",
            xytext=(-50, -70),
            textcoords="offset points",
            arrowprops=dict(facecolor="black", shrink=0.05),
            bbox=dict(boxstyle="square,pad=50", fc="none", ec="none"),
        )

        plt.axvspan(0, x1_max, alpha=0.3, color="gray")
        plt.legend()

        plt.xlabel("$T$", labelpad=20)
        plt.ylabel("$M$", labelpad=20)

        title = "$l = {}$\n$P = {}$, $c_{{t/m}} = {}$".format(lang, x1_max, c12)
        plt.title(title)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plot_file = os.path.join(
            save_dir, f"expansion_path_{lang}_{x1_max}_{str(c12).replace('.','')}.pdf"
        )
        plt.savefig(plot_file, bbox_inches="tight")

    return iso_perf_points_df, iso_cost_points_df, op_points
