#!/usr/bin/env python

# -*- coding: utf-8 -*-

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd


def get_target_shapes() -> pd.DataFrame:
    # Basic data: textual description, M, N, K
    df: pd.DataFrame = pd.DataFrame(
        [
            ("Shape #1", 20196, 512, 1536),
            ("Shape #2", 171792, 512, 1536),
            ("Shape #3", 173318, 512, 1536),
        ],
        columns=["desc", "M", "N", "K"],
    )
    # Compute arithmetic intensity:
    M: pd.Series = df["M"]
    N: pd.Series = df["N"]
    K: pd.Series = df["K"]
    ops: pd.Series = 2 * M * N * K
    bytes: pd.Series = 2 * M * K + 2 * K * N + 4 * M * N
    df["arith_intensity"] = ops / bytes
    return df


def get_target_hardware_specs() -> pd.DataFrame:
    # Basic data: textual description, compute performance in TOp/s, memory bandwidth in TB/s
    df: pd.DataFrame = pd.DataFrame(
        [
            ("Peak MI300X", 1307.4, 5.3),
            ("Achievable MI300X", 650, 3.5),
            ("Peak H100 PCIe", 1513, 2),
            ("Peak H100 SXM", 1979, 3.35),
        ],
        columns=["desc", "comp_perf_tops", "mem_bandwidth_tbs"],
    )
    # Compute arithmetic ops : bytes ratio:
    # We can divide without converting units because both compute and memory capabilities are
    # expressed with Tera prefix.
    df["ops_bytes_ratio"] = df["comp_perf_tops"] / df["mem_bandwidth_tbs"]
    return df


class ColorPalette:
    colors: list[str]
    color_index: int

    def __init__(self):
        # Categorical color palette from this website:
        # https://carbondesignsystem.com/data-visualization/color-palettes/#categorical-palettes
        self.colors = [
            "#8a3ffc",  # Purple 60
            "#33b1ff",  # Cyan 40
            "#007d79",  # Teal 60
            "#ff7eb6",  # Magenta 40
            "#fa4d56",  # Red 50
            "#6fdc8c",  # Green 30
            "#4589ff",  # Blue 50
            "#d12771",  # Magenta 60
            "#d2a106",  # Yellow 40
            "#08bdba",  # Teal 40
            "#bae6ff",  # Cyan 20
            "#ba4e00",  # Orange 60
            "#d4bbff",  # Purple 30
        ]
        self.color_index = 0

    def next_color(self) -> str:
        color: str = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        return color


def main() -> None:
    # Get data:
    shapes_df: pd.DataFrame = get_target_shapes()
    hw_df: pd.DataFrame = get_target_hardware_specs()

    # Set chart size:
    fig_width: float
    fig_height: float
    fig_width, fig_height = plt.rcParams["figure.figsize"]
    fig_scale: float = math.sqrt(2)
    fig_width *= fig_scale
    fig_height *= fig_scale
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

    # Set chart title and axes' titles:
    # TODO: Improve formatting of titles.
    ax.set(
        title="Phantom addmm Kernel Roofline",
        xlabel="Arithmetic Intensity (Op / B)",
        ylabel="Performance (TOp / s)",
    )

    # X-axis is arithmetic intensity:
    max_x: float = 1.05 * max(shapes_df["arith_intensity"].max(), hw_df["ops_bytes_ratio"].max())
    x: npt.NDArray = np.arange(0, max_x, 0.1)

    # Get color palette:
    color_palette: ColorPalette = ColorPalette()

    # Plot rooflines for each hardware:
    rooflines: list = []
    for _, hw in hw_df.iterrows():
        y: npt.NDArray = np.minimum(hw["comp_perf_tops"], hw["mem_bandwidth_tbs"] * x)
        color: str = color_palette.next_color()
        roofline, = ax.plot(x, y, label=hw["desc"], color=color)
        rooflines.append(roofline)
        ax.vlines(
            x=hw["ops_bytes_ratio"],
            ymin=0,
            ymax=hw["comp_perf_tops"],
            color=color,
            linestyles="dotted",
        )

    # Plot legend for rooflines:
    rooflines_legend = ax.legend(handles=rooflines, title="Hardware Rooflines", loc="upper left",
                                 bbox_to_anchor=(1.05, 1))
    ax.add_artist(rooflines_legend)

    # Plot arithmetic intensity of each shape:
    max_y: float = ax.get_ylim()[1]
    shapes: list = []
    for _, shape in shapes_df.iterrows():
        # TODO:
        # * Improve shape legend text, i.e. display (M, N, K).
        shape = ax.vlines(
            x=shape["arith_intensity"],
            ymin=0,
            ymax=max_y,
            label=shape["desc"],
            color=color_palette.next_color(),
            linestyles="dashed",
        )
        shapes.append(shape)

    # Plot lengend for shapes:
    roofline_legend_bbox = rooflines_legend.get_window_extent().transformed(fig.transFigure.inverted())
    shapes_legend = ax.legend(
        handles=shapes,
        title="Shapes",
        loc="upper left",
        bbox_to_anchor=(1.05, roofline_legend_bbox.y0),
    )
    ax.add_artist(shapes_legend)

    # Save chart to image file:
    plot_img_file_name: str = os.path.splitext(os.path.basename(__file__))[0] + ".png"
    fig.savefig(plot_img_file_name)


if __name__ == "__main__":
    main()
