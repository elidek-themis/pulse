import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

STAR = {
    "xdata": [0],
    "ydata": [0],
    "marker": "*",
    "color": "black",
    "markersize": 6,
    "linestyle": "None",
    "label": "actual difference",
}

CIRCLE = {
    "xdata": [0],
    "ydata": [0],
    "marker": "o",
    "color": "black",
    "markersize": 4,
    "linestyle": "None",
    "label": "predicted difference",
}


def lineplot(diff, figsize=(8, 6), group_a_color="blue", group_b_color="red") -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    sns.pointplot(
        data=diff,
        x="$diff$",
        y="index",
        hue="mean",
        dodge=False,
        palette={"Group A": group_a_color, "Group B": group_b_color},
        linestyle="none",
        markersize=5,
        linewidth=2,
        seed=2025,
    )

    if "pct_diff" in diff:
        ax.scatter(
            data=diff,
            x="pct_diff",
            y="index",
            marker="*",
            c="black",
            edgecolors="black",
            linewidths=0.8,
            s=45,
            alpha=0.5,
            zorder=2,
        )

    _, fig_height = figsize
    ax.set_ylabel("")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.legend(
        handles=[Line2D(**CIRCLE), Line2D(**STAR)],
        loc="upper center",
        bbox_to_anchor=(0.475, 1 + 0.075 * (7 / fig_height)),
        ncols=2,
    )
    ax.set_xticks([-1, 0, 1])

    return fig
