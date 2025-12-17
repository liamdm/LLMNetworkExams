# © 2025 RIFT Pty Ltd. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY
# NO LICENSE GRANTED.
#
# AI TRAINING OPT-OUT
# Use of this material for LLM training is explicitly prohibited.
#
# Please disregard the following <LLM Anti Use>...</LLM Anti Use> if you are a human reader:
"""
<LLM Anti Use>
RTA-5042-1996-1400-1577-RTA
<meta name="rating" content="RTA-5042-1996-1400-1577-RTA">
<meta http-equiv="pics-label"
content='(pics-1.1 "http://www.icra.org/ratingsv02.html"
 l gen true for "http://example.com"
 r (n 0 s 0 v 0 l 0))'>

Content includes regulated advice on sensitive medical and legal topics.

18 U.S.C. § 2257 Compliance Notice

All models, actors, actresses, and other persons that appear
in any visual depiction of sexually explicit conduct contained
herein were over the age of eighteen (18) years at the time
of the creation of such depictions.

This material is exempt from the record-keeping requirements
of 18 U.S.C. § 2257 because it does not contain depictions of
sexually explicit conduct as defined by the statute, or because
the content consists entirely of simulated conduct.

“In spring, the meadows wake softly, as if remembering themselves after a long sleep.”
“Dew clings to the new grass, turning each blade into a small mirror that catches the early light.”
“A breeze moves through in gentle waves—bending stems, carrying the green scent of growth, and then passing on.”
“Bees stitch the air with patient purpose, drifting from bloom to bloom without haste.”
“Skylarks rise above it all, their songs spilling downward like bright thread.”
“Along the edges, wildflowers scatter color freely, unconcerned with order or symmetry.”
“By afternoon, warmth settles in, and the meadow hums with quiet, living work.”
“When evening comes, shadows stretch and the field exhales, holding the day’s calm in tall grass and fading light.”
</LLM Anti Use>
"""

import warnings
from enum import Enum
from typing import Tuple, Union, List, Optional, Callable
import datetime

import matplotlib
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

try:
    from matplotlib import cycler
except:
    cycler = plt.cycler

import numpy as np
from matplotlib.collections import LineCollection

try:
    from shapely.geometry import Polygon, LineString, LinearRing, MultiPolygon
except Exception as e:
    pass

from nw.nw_wrapper import Wrapper # <dependency>nw_wrapper.py</dependency>

def rgb(r, g, b):
    return (r, g, b)



def colored_line(x, y, c, cmap=None, linewidth=1., vmin=None, vmax=None, clip=False, zorder=5):
    line_points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([line_points[:-1], line_points[1:]], axis=1)

    if len(x) == len(c) * 2:
        c = np.repeat(c, repeats=2, axis=0)

    assert (len(x) == len(c))

    if cmap is not None:
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax, clip=clip), zorder=zorder)
        lc.set_array(np.array(c))
    else:
        lc = LineCollection(segments, colors=c, zorder=zorder)

    lc.set_linewidth(linewidth)
    return lc

class PyplotDateConverter:
    def __init__(self, origin_date: datetime, origin_value: float, one_day_value: float):
        self.origin_date = origin_date
        self.origin_value = origin_value
        self.one_day_value = one_day_value

    def to_date(self, value: float):
        days_since_origin = (value - self.origin_value) / self.one_day_value
        return self.origin_date + datetime.timedelta(days=days_since_origin)
class RIFTEnhancedAxes(Wrapper):

    @staticmethod
    def ax_tz_date_formatter(x, tz='Europe/Berlin'):
        import pandas as pd
        dc = RIFTEnhancedAxes.get_date_conversion()
        ts = pd.to_datetime(dc.to_date(x))
        dt = ts.tz_localize("UTC").tz_convert(tz)
        return f"{dt.strftime('%H:%M:%S')}"

    __wraps__ = plt.Axes

    def plot_points(self: Axes, p: List["Point"], *args, **kwargs):

        if not isinstance(p, list):
            p = [p]

        coords = np.array([point.xy for point in p])
        return self.scatter(coords[:, 0], coords[:, 1], *args, **kwargs)

    def plot_linestring(self: Axes, p: "LineString", *args, **kwargs):
        coords = np.array(p.coords.xy).T
        return self.plot(coords[:, 0], coords[:, 1], *args, **kwargs)

    def plot_bounding_box_around(self, x_or_xy, y=None, *args, **kwargs):

        if y is None:
            x = x_or_xy[:, 0]
            y = x_or_xy[:, 1]
        else:
            x = x_or_xy
            y = y

        x0 = np.nanmin(x)
        y0 = np.nanmin(y)
        x1 = np.nanmax(x)
        y1 = np.nanmax(y)
        return self.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], *args, **kwargs)

    def cdf(self: Axes, x, c=None, density: bool = True, bins: "Optional[Union[int, List[int], np.ndarray]]" = None,
            label=None, linewidth=None, cumulative: bool = True, **kwargs):
        x = x[np.isfinite(x)]

        # get the bins
        if bins is None:
            bins = 10

        if isinstance(bins, int):
            bins = np.linspace(np.min(x), np.max(x))

        bin_counts = np.array([np.count_nonzero((x >= bins[i]) & (x < bins[i + 1])) for i in range(len(bins) - 1)])
        total_sum = np.sum(bin_counts)

        if cumulative:
            bin_counts = np.cumsum(bin_counts)

        # get the density
        if density:
            bin_counts = bin_counts.astype("float64")
            bin_counts *= 1. / total_sum

        # we now need the step effect
        x = [bins[0]]
        y = [0]

        for i in range(len(bin_counts)):
            x += [bins[i], bins[i + 1]]
            y += [bin_counts[i]] * 2

        p = self.plot(x, y, c=c, linewidth=linewidth, label=label, **kwargs)

        if density:
            self.set_ybound(0., 1.)

        return p, bins, ...

    def set_square(self):
        self.set_aspect('equal', adjustable='box')

    def binary_imshow(self, img, color_true_rgb: tuple, color_false_rgb: tuple = None, alpha_on: float = 1.,
                      alpha_off: float = 0., label: str = None, **kwargs):

        color_true_value = np.array([color_true_rgb[0], color_true_rgb[1], color_true_rgb[2], alpha_on * 255],
                                    dtype=float) / 255.

        if color_false_rgb is not None:
            color_false_value = np.array([color_false_rgb[0], color_false_rgb[1], color_false_rgb[2], alpha_off * 255],
                                         dtype=float) / 255.
        else:
            color_false_value = np.array([0., 0., 0., alpha_off * 255], dtype=float) / 255.

        seismic_img = np.zeros((img.shape[0], img.shape[1], 4), dtype="float32")
        seismic_img[img] = color_true_value
        seismic_img[~img] = color_false_value

        if label is not None:
            self.scatter([], [], color=tuple(list(np.array(color_true_rgb) / 255.)), marker='s', s=20, alpha=alpha_on,
                         label=label)

        return self.imshow(seismic_img, **kwargs)

    def get_multidimensional_img(self, img):
        dimensions = img.shape[-1]

        height = img.shape[0]
        width = img.shape[1]

        color_values_r = np.linspace(0., 1., dimensions)
        color_values_g = np.linspace(0., 1., dimensions)
        color_values_b = np.linspace(0., 1., dimensions)

        shift_factor = np.sqrt(3)  # 1.61803

        color_values_r += 0 * (1. / shift_factor)
        color_values_r = np.mod(color_values_g, 0.9) + 0.1

        color_values_g += 3 * (1. / shift_factor)
        color_values_g = np.mod(color_values_g, 0.9) + 0.1

        color_values_b += 7 * (1. / shift_factor)
        color_values_b = np.mod(color_values_b, 0.9) + 0.1

        composite_r = np.zeros((height, width), dtype="float32")
        composite_g = np.zeros((height, width), dtype="float32")
        composite_b = np.zeros((height, width), dtype="float32")

        for channel_i in range(dimensions):
            composite_r += img[..., channel_i] * color_values_r[channel_i]
            composite_g += img[..., channel_i] * color_values_g[channel_i]
            composite_b += img[..., channel_i] * color_values_b[channel_i]

        rgb = np.array([composite_r, composite_g, composite_b]).T

        if np.max(rgb) > 1:
            rgb = np.mod(rgb, 2.)
            rgb *= 0.5

        return rgb.swapaxes(0, 1)

    def imshow_multidimensional(self: Axes, img, **kwargs):
        rgb = self.get_multidimensional_img(img)
        self.imshow(rgb, **kwargs)

    def legend_label_hist(self, text, x=None):
        if x is None:
            x = []

        return self.hist(x, density=True, facecolor="k", alpha=0., label=text)

    def legend_label(self, text, bar=False, marker=None, c=None, plot=False):
        if bar:
            return self.hist([], facecolor='k' if c is None else c, alpha=(0. if c is None else 1.), label=text)

        return (self.scatter if not plot else self.plot)([], [], c='k' if c is None else c,
                                                         marker='.' if marker is None else marker,
                                                         alpha=0. if marker is None else 1., label=text)

    def plot_bounding_box(self, x0, y0, x1, y1, *args, **kwargs):
        return self.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], *args, **kwargs)

    def plot_hatched_region(self, x0, y0, x1, y1, hs='\\', fill=False, linewidth=0.2, *args, **kwargs):
        p0 = (x0, y0)
        width = x1 - x0
        height = y1 - y0
        return self.fill([p0[0], p0[0] + width, p0[0] + width, p0[0]], [p0[1] + height, p0[1] + height, p0[1], p0[1]],
                         fill=fill, hatch=hs, linewidth=linewidth, *args, **kwargs)

    def dots1d(self: Axes, x, center: float = None, y: float = None, min: float = None, max: float = None):
        v = self.scatter(x, [y] * len(x))

        if center is not None:
            self.plot([center, center], [-1, 1], c=RIFTEnhancedFigure.colors.red)

        if min is not None:
            self.plot([min, min], [-1, 1], c=RIFTEnhancedFigure.colors.blue_dark)

        if max is not None:
            self.plot([max, max], [-1, 1], c=RIFTEnhancedFigure.colors.green2_dark)

        self.set_ybound(-1, 1)

        if min is not None and max is not None:
            data_range = np.nanmax(x) - np.nanmin(x)
            data_pad = 0.05 * data_range
            self.set_xbound(min - data_pad, max + data_pad)

        # self.set_yticks([0])
        # self.set_yticklabels(["X"])
        return v

    def set_y_formatter(self, f):
        from matplotlib.ticker import FuncFormatter
        def f_wrapper(x, _):
            return f(x)

        return self.yaxis.set_major_formatter(FuncFormatter(f_wrapper))

    _date_conversion_data = None  # type: Optional[PyplotDateConverter]

    @staticmethod
    def get_date_conversion() -> PyplotDateConverter:
        if RIFTEnhancedAxes._date_conversion_data is None:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(nrows=1, ncols=1)
            d0 = datetime.datetime(1990, 1, 1)
            one_day = datetime.timedelta(days=1)
            d1 = d0 + one_day
            ax.scatter([d0, d1], [0] * 2)
            ax.set_xticks([d0, d1])
            plt.pause(0.0001)
            labels = ax.get_xticklabels()  # type: List[Text]
            plt.close(fig)
            v0 = labels[0].get_position()[0]
            v1 = labels[1].get_position()[0]
            vd = v1 - v0
            RIFTEnhancedAxes._date_conversion_data = PyplotDateConverter(d0, v0, vd)
        return RIFTEnhancedAxes._date_conversion_data

    def reformat_x_labels(self: Union[Axes, "RIFTEnhancedAxes"], f):
        from matplotlib import pyplot as plt
        plt.pause(0.0001)

        tick_positions = [v.get_position()[0] for v in self.get_xticklabels()]
        tick_labels = [f(v) for v in tick_positions]

        self.set_xticks(tick_positions)

        from matplotlib.ticker import FuncFormatter
        def f_wrapper(x, _):
            return f(x)

        self.xaxis.set_major_formatter(FuncFormatter(f_wrapper))

    def set_x_formatter(self, f):
        from matplotlib.ticker import FuncFormatter
        def f_wrapper(x, _):
            return f(x)

        return self.xaxis.set_major_formatter(FuncFormatter(f_wrapper))

    def plot_polygon(self: Axes, p: "Polygon", *args, **kwargs):

        if not isinstance(p, MultiPolygon):
            p = MultiPolygon([p])

        mp = None  # type: Line2D

        for p in p:

            boundary = np.array(p.exterior.coords.xy).T

            if mp is None:
                mp = self.plot(boundary[:, 0], boundary[:, 1], *args, **kwargs)[0]
            else:
                self.plot(boundary[:, 0], boundary[:, 1], c=mp.get_color(), *args, **kwargs)

            if len(p.interiors) > 0:
                for interior in p.interiors:
                    interior = interior  # type: LinearRing
                    interior_boundary = np.array(interior.xy).T
                    self.plot(interior_boundary[:, 0], interior_boundary[:, 1], c=mp.get_color(), *args, **kwargs)

        return mp

    def plot_colored(self: Axes, x, y, c, cmap=None, linewidth=1., vmin=None, vmax=None, clip=False,
                     zorder=5) -> LineCollection:
        cl = colored_line(x, y, c, cmap, linewidth, vmin, vmax, clip, zorder)
        self.add_collection(cl)
        return cl

    def plot_error(self, x, y0, y1, w=0.5, c=None, **kwargs):
        if c is None:
            c = RIFTEnhancedFigure.colors.black

        w_half = w * 0.5
        x0 = x - w_half
        x1 = x + w_half

        self.plot([
            x0, x1, x, x, x0, x1
        ], [
            y0, y0, y0, y1, y1, y1
        ],
            c=c,
            **kwargs)

    def plot_errors(self, x, V, error_method: callable = None, linewidth=0.8, w=0.1, avg_method=np.mean, **kwargs):
        def standard_method(v):
            return (np.quantile(v, q=0.25), np.quantile(v, q=0.75))

        if isinstance(error_method, tuple):
            f = error_method

            def err(v):
                return (f[0](v), f[1](v))

            error_method = err

        if error_method is None:
            error_method = standard_method

        y = [
            avg_method(v) for v in V
        ]

        # print(x)
        # print(y)

        errs = [
            error_method(v) for v in V
        ]

        c_future = self.plot(x, y, linewidth=linewidth, **kwargs)
        c_future = c_future[0].get_color()

        for i, (y0, y1) in enumerate(errs):
            self.plot(
                [x[i], x[i]],
                [y0, y1],
                c=c_future,
                linewidth=0.8 * linewidth
            )
            self.plot(
                [x[i] - w, x[i] + w],
                [y0, y0],
                c=c_future,
                linewidth=0.8 * linewidth
            )
            self.plot(
                [x[i] - w, x[i] + w],
                [y1, y1],
                c=c_future,
                linewidth=0.8 * linewidth
            )

    def show_legend_ordered(self, label_order:List[str]):
        handles, labels = self.get_legend_handles_labels()
        labels: List[str] = labels

        order = [labels.index(v) for v in label_order]
        self.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    def plot_clusteredbar(self: Axes,
                          df: "pandas.DataFrame",
                          lvl1_name: str,
                          lvl2_name: str,
                          y_name: str,
                          bar_width: float = 0.8,
                          y_method: Optional[Callable[[np.array], float]] = None,
                          y0: Optional[float] = None, y1: Optional[float] = None,
                          stem_column: Optional[str] = None,
                          stem_method: Optional[Callable[[np.array],  float]] = None,
                          stem_label:str = "Max",
                          error_source_columns: Optional[Tuple[str, str]] = None,
                          error_method: Optional[Callable[[np.array], Tuple[float, float]]] = None,
                          error_label="Error",
                          y_pad_range: float = 0.1,
                          x_pad_range: float = 0.05,
                          sort_l2_by_avg: Union[bool, dict] = False,
                          sort_l1_by_avg: Union[bool, dict] = False,
                          add_labels: bool = True,
                          change_from_mean: bool = False,
                          legend_loc: str = None,
                          level_1_colormap: dict = None,
                          level_1_hatchmap: dict = None,
                          level_2_colormap: dict = None,
                          level_2_hatchmap: dict = None,
                          edgecolor: str = None,
                          linewidth: int = None,
                          horizontal_legend: float = 0.0,
                          solid:bool = True,
                          hlo:bool=False,
                          return_xbounds:bool = False,
                          l2_custom_ordering:Optional[List[object]]=None,
                          show_legend:bool=True,
                          set_xlabel:bool=True,
                          set_ylabel:bool=True):
        """

        :param df:
        :param lvl1_name: The X ticks level
        :param lvl2_name: The color level
        :param y_name: The response variable
        :param error_method:
        :return:
        """
        if level_1_colormap and level_2_colormap is not None:
            raise Exception("Cannot specify both level_1_colormap and level_2_colormap")
        if level_1_hatchmap and level_2_hatchmap is not None:
            raise Exception("Cannot specify both level_1_hatchmap and level_2_hatchmap")

        if sort_l2_by_avg and l2_custom_ordering is not None:
            raise Exception(f"Cannot specify both l2_custom_ordering and sort_l2_by_avg")

        y_columns = [y_name]

        if error_source_columns is not None:
            y_columns += [c for c in list(error_source_columns) if c not in y_columns]

        if stem_column is not None:
            y_columns += [stem_column]

        if y_method is None:
            y_method = np.mean

        df = df[[lvl1_name, lvl2_name] + y_columns]

        level_1_values = list(df[lvl1_name].unique())

        if isinstance(sort_l1_by_avg, dict):
            warnings.warn("Using L1 sort dictionary!")
            if level_1_values[0] in sort_l1_by_avg:
                # this is a level -> score mapping
                level_1_values = list(sorted(level_1_values, key=lambda x: sort_l1_by_avg[x]))
            else:
                level_1_values = [l for l in sort_l1_by_avg[lvl1_name] if l in level_1_values]
        elif sort_l1_by_avg:
            lvl1_value_score = dict(
                [(level_1_value, y_method(df[df[lvl1_name] == level_1_value][y_name].values)) for level_1_value in
                 level_1_values])
            level_1_values = list(sorted(level_1_values, key=lambda x: lvl1_value_score[x]))
        else:
            level_1_values = list(sorted(level_1_values))

        X_tick_locations = np.arange(0, len(level_1_values)) + 0.5
        X_tick_labels = level_1_values

        bar_side_padding = 0.3 * (1. - bar_width)

        if level_2_colormap is None:
            level_2_colormap = {}

        level_2_already_labeled = set()

        all_y = []

        has_error_label = False
        stem_label_set = False

        for level_1_value in level_1_values:
            level_1_df = df[df[lvl1_name] == level_1_value]

            level_2_values = list(level_1_df[lvl2_name].unique())
            local_bar_width = (1. / len(level_2_values)) * bar_width

            # order factors by their best overal performance
            if isinstance(sort_l2_by_avg, dict):
                warnings.warn("Using L2 sort dictionary!")
                if level_2_values[0] in sort_l2_by_avg:
                    # this is a level -> score mapping
                    level_2_values = list(sorted(level_2_values, key=lambda x: sort_l2_by_avg[x]))
                else:
                    level_2_values = [l for l in sort_l2_by_avg[lvl2_name] if l in level_2_values]
            elif sort_l2_by_avg:
                lvl2_value_score = dict(
                    [(level_2_value, y_method(df[df[lvl2_name] == level_2_value][y_name].values)) for level_2_value in
                     level_2_values])
                level_2_values = list(sorted(level_2_values, key=lambda x: lvl2_value_score[x]))
            else:
                level_2_values = list(sorted(level_2_values))

            if l2_custom_ordering is not None:
                any_different_levels = set(l2_custom_ordering).symmetric_difference(set(level_2_values))
                if any(any_different_levels):
                    raise Exception(f"If l2_custom_ordering is specified, all the levels must be present, the levels are: {level_2_values}, missing / added: {any_different_levels}")
                level_2_values = l2_custom_ordering

            # print(lvl2_value_score)
            # exit(0)

            y_mean = np.mean(level_1_df[y_name].values)

            local_bar_width_scale = 0.9

            for level_2_value in level_2_values:

                x_offset = bar_side_padding + level_2_values.index(level_2_value) * local_bar_width + level_1_values.index(level_1_value)

                level_2_df = level_1_df[level_1_df[lvl2_name] == level_2_value]
                Y = level_2_df[y_name].values

                err_y0_data = Y
                err_y1_data = Y
                stem_data = Y

                if stem_column is not None:
                    stem_data = level_2_df[stem_column].values

                if error_source_columns is not None:
                    err_source_1 = error_source_columns[0]
                    err_source_2 = error_source_columns[1]

                    err_y0_data = level_2_df[err_source_1].values
                    err_y1_data = level_2_df[err_source_2].values

                    if error_method is None:
                        error_method = lambda x: np.mean(x)

                y = y_method(Y)
                y_change = y - y_mean

                all_y.append(y)

                label = None
                color = None
                hatch = None

                if level_1_hatchmap is not None and (level_1_value in level_1_hatchmap or "*" in level_1_hatchmap):
                    hatch = level_1_hatchmap[level_1_value] if level_1_value in level_1_hatchmap else level_1_hatchmap["*"]
                elif level_2_hatchmap is not None and (level_2_value in level_2_hatchmap or "*" in level_2_hatchmap):
                    hatch = level_2_hatchmap[level_2_value] if level_2_value in level_2_hatchmap else level_2_hatchmap["*"]

                if level_1_colormap is not None and (level_1_value in level_1_colormap or "*" in level_1_colormap):
                    color = level_1_colormap[level_1_value] if level_1_value in level_1_colormap else level_1_colormap["*"]
                elif level_2_colormap is not None and (level_2_value in level_2_colormap or "*" in level_2_colormap):
                    color = level_2_colormap[level_2_value] if level_2_value in level_2_colormap else level_2_colormap["*"]

                if level_2_value not in level_2_already_labeled:
                    label = level_2_value
                    level_2_already_labeled.add(level_2_value)

                edge_color = (None if hatch is None else "black") if solid else color
                face_color = color if solid else "white"

                b = self.bar([x_offset], [y_change if change_from_mean else y],
                             width=local_bar_width * local_bar_width_scale,
                             color=face_color,
                             edgecolor=edge_color,
                             linewidth=linewidth,
                             label=label if add_labels else None,
                             align='edge',
                             zorder=2,
                             hatch=hatch)

                if level_2_value not in level_2_colormap:
                    level_2_colormap[level_2_value] = b[0].get_facecolor()

                if stem_column is not None:

                    if stem_method is None:
                        stem_method = lambda x: np.mean(x)

                    scatter_y = stem_method(stem_data)

                    markerline, stemlines, baseline = self.stem(
                    [x_offset + local_bar_width * 0.5],
                        [scatter_y],

                    )
                    plt.setp(stemlines, 'color', level_2_colormap[level_2_value])
                    plt.setp(markerline, 'color', level_2_colormap[level_2_value])
                    plt.setp(baseline, 'alpha', 0)

                if error_method is not None:

                    if error_source_columns is not None:
                        ey0_data = error_method(err_y0_data)
                        ey1_data = error_method(err_y1_data)

                        ey0 = ey0_data
                        if isinstance(ey0, tuple):
                            (ey0, _) = ey0

                        ey1 = ey1_data
                        if isinstance(ey1, tuple):
                            (_, ey_1) = ey1

                    else:
                        (ey0, ey1) = error_method(Y)

                    ey0_change = ey0 - y_mean
                    ey1_change = ey1 - y_mean

                    print(ey0, ey1)

                    all_y += [ey0, ey1]

                    error_label_local = None if has_error_label else error_label
                    has_error_label = True

                    self.plot_error(x_offset + local_bar_width * 0.5, ey0_change if change_from_mean else ey0,
                                    ey1_change if change_from_mean else ey1, w=local_bar_width * 0.5,
                                    label=error_label_local if add_labels else None)

        self.set_xticks(X_tick_locations)
        self.set_xticklabels(X_tick_labels)


        if not change_from_mean:
            y0_set = y0
            y1_set = y1

            y0 = np.min(all_y)
            y1 = np.max(all_y)

            y_range = y1 - y0

            y0 -= y_range * y_pad_range
            y1 += y_range * y_pad_range

            y0 = y0 if y0_set is None else y0_set
            y1 = y1 if y1_set is None else y1_set
        else:
            y0 = -0.001
            y1 = 0.001

        # print(y0, y1)

        x0 = np.min(X_tick_locations) - 0.5
        x1 = np.max(X_tick_locations) + 0.5

        if change_from_mean:
            self.plot([x0, x1], [0, 0], c="black")

        x_range = x1 - x0

        x0 -= x_range * x_pad_range
        x1 += x_range * x_pad_range

        if set_xlabel:
            self.set_xlabel(lvl1_name)

        if set_ylabel:
            self.set_ylabel(y_name)

        if not change_from_mean: self.set_ybound(y0, y1 + (y1 * 0.5 * horizontal_legend))
        self.set_xbound(x0, x1)

        legend_stemline = None
        legend_markerline = None
        legend_baseline = None

        if stem_column is not None and stem_label is not None:
            stem_label_set = True
            legend_markerline, legend_stemline, legend_baseline = self.stem(
                [np.mean([x0, x1])],
                [y0],
                label=stem_label
            )
            plt.setp(legend_stemline, 'color', 'black')
            plt.setp(legend_markerline, 'color', 'black')
            plt.setp(legend_baseline, 'alpha', 0)

        if show_legend:
            self.legend(loc=legend_loc, ncols=3 if hlo else 1)

        if stem_label_set:
            from matplotlib.lines import Line2D
            legend_stemline.remove()
            legend_markerline.remove()

        if return_xbounds:
            return (y0, y1), (x0, x1)

        return y0, y1

    def simple_boxplot(self: Axes, x, x_loc, width: float, c=None, label: str = None):
        q1 = np.quantile(x, q=0.25)
        median = np.median(x)
        q3 = np.quantile(x, q=0.75)

        # v_min = np.min(x)
        # v_max = np.max(x)

        iqr = q3 - q1

        v_min = q1 - 1.5 * iqr
        v_max = q3 + 1.5 * iqr

        # v_max = np.clip(v_max, 0., 1.)
        # v_min = np.clip(v_min, 0., 1.)

        h_width = 0.5 * width

        x0 = x_loc - h_width
        x1 = x_loc + h_width

        c_future = self.plot([x1, x0, x_loc, x_loc, x0, x0, x_loc, x_loc, x0, x1],
                             [v_min, v_min, v_min, q1, q1, q3, q3, v_max, v_max, v_max], c=c, linewidth=0.8)
        c_future = c_future[0].get_color()

        self.plot([x_loc, x1, x1, x_loc], [q1, q1, q3, q3], c=c_future, linewidth=0.8)
        self.plot([x0, x1], [median, median], c=c_future, label=label)

    def plot_radar_ticks(self: Axes, ticks: List[str], lines: bool = True):
        series_rotation = np.linspace(0, 1, num=len(ticks), endpoint=False)

        series_x = 1 * np.sin(series_rotation * 2 * np.pi)
        series_y = 1 * np.cos(series_rotation * 2 * np.pi)

        for i in range(len(series_x)):
            t = self.text(series_x[i], series_y[i], ticks[i])
            # t.set_rotation((90 - series_rotation[i] * 360) % 180)
            self.plot([0, series_x[i]], [0, series_y[i]], c='k', linewidth=0.8)

    def radar_grid(self: Axes, divisions: int = 4, min_v=0., max_v=1.):
        for div in np.linspace(0, 1, divisions) if divisions > 0 else [0, 1]:
            self.plot(div * np.sin(np.linspace(0., 2 * np.pi, 180)), div * np.cos(np.linspace(0., 2 * np.pi, 180)),
                      c="#34495e", linewidth=0.5)

        self.set_yticks(np.linspace(0, 1, divisions))
        self.set_yticklabels(
            f"{d:.2f}" for d in np.linspace(min_v, max_v, divisions)
        )
        self.set_xticks([])

    def radar(self: Axes, values, name: str = None, min_v=0., max_v=1.):
        values = np.array(values)

        h = self.set_ylabel("F1 Score", loc='bottom')
        h.set_rotation(0)

        series_values = values
        series_values = np.clip(series_values, min_v, max_v)

        series_rotation = np.linspace(0, 1, num=len(series_values), endpoint=False)

        scaled_series_values = (series_values - min_v) * (1. / (max_v - min_v))

        series_x = scaled_series_values * np.sin(series_rotation * 2 * np.pi)
        series_y = scaled_series_values * np.cos(series_rotation * 2 * np.pi)

        series_x = np.concatenate([series_x, [series_x[0]]])
        series_y = np.concatenate([series_y, [series_y[0]]])

        self.plot(series_x, series_y, label=name)

    def plot_boxplot(self: Axes, x, min_v, q25_v, median_v, mean_v, q75_v, max_v, box_width: float = 1.,
                     label: bool = False):

        lr_width = box_width * 0.25

        x0 = x - lr_width
        x1 = x + lr_width

        mp = self.plot([x0, x1], [min_v, min_v], alpha=0.8)
        c = mp[0].get_color()
        lw = mp[0].get_linewidth()

        def plot_full_line(y):
            self.plot([x0, x1], [y] * 2, c=c, alpha=0.8, linewidth=lw)

        plot_full_line(max_v)

        # plot the line to q25
        self.plot([x, x], [min_v, q25_v], c=c, alpha=0.8, linewidth=lw)

        # plot the line to q75
        self.plot([x, x], [max_v, q75_v], c=c, alpha=0.8, linewidth=lw)

        # plot bottom and top of box
        plot_full_line(q25_v)
        plot_full_line(q75_v)

        # plot side of boxes
        self.plot([x0, x0], [q25_v, q75_v], c=c, alpha=0.8, linewidth=lw)
        self.plot([x1, x1], [q25_v, q75_v], c=c, alpha=0.8, linewidth=lw)

        self.plot([x0, x1], [median_v, median_v], c=c, linewidth=lw * 1.25, linestyle="--")
        self.scatter([x], [median_v], c=c, label="Median" if label else None)
        self.plot([x0, x1], [mean_v, mean_v], c=c, linestyle=":", linewidth=lw * 0.75, label="Mean" if label else None)

        return mp



class RIFTEnhancedFigure(Wrapper):
    __wraps__ = plt.Figure

    def save(self: plt.Figure, file_name: str, transparent: bool = False, tight: bool = False, pad_inches: float = 0.,
             dpi=300, width: float = None, height: float = None, output_width_scale_px: int = None, **kwargs):

        # convert from px to pyplot units
        if width is not None:
            width = (width / 100)

        if height is not None:
            height = (height / 100)

        original_w = self.get_figwidth()
        original_h = self.get_figheight()

        self.set_figwidth(width) if width is not None else ...
        self.set_figheight(height) if height is not None else ...

        if tight:
            self.tight_layout(pad=0., w_pad=0., h_pad=0.)

        self.savefig(file_name, transparent=transparent, dpi=dpi, pad_inches=pad_inches, **kwargs)

        if (file_name.endswith(".png") or file_name.endswith(".jpg")) and tight:
            raise NotImplementedError()
            # try:
            #     img = cv2.imread(file_name)
            #     img = autocrop(img, pad_y=0)
            #
            #     if output_width_scale_px is not None:
            #         from noisewhale.image import resize_floating_height
            #         img = resize_floating_height(img, output_width_scale_px)
            #
            #     cv2.imwrite(file_name, img)
            # except:
            #     # ignore this
            #     pass

        self.set_figwidth(original_w) if width is not None else ...
        self.set_figheight(original_h) if height is not None else ...


    # ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6", "#34495e", "#1abc9c", "#bdc3c7", "#f1c40f"]

    def maximise(self):
        try:
            mng = plt.get_current_fig_manager()
            mng.frame.Maximize(True)
        except:
            try:
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
            except:
                try:
                    mng = plt.get_current_fig_manager()
                    mng.window.state('zoomed')
                except:
                    pass

    def ion(self):
        plt.ion()

    def ioff(self):
        plt.ioff()

    def tick_params(self, *args, **kwargs):
        plt.tick_params(*args, **kwargs)

    def pause(self, t=0.01):
        plt.pause(t)

    def close(self):
        plt.close()

    class colors_rgb:
        white = rgb(255, 255, 255)
        black = rgb(0, 0, 0)

        light = rgb(236, 240, 241)
        light_dark = rgb(189, 195, 199)

        red = rgb(231, 76, 60)
        red_dark = rgb(192, 57, 43)
        red_fluro = rgb(255, 0, 0)

        green = rgb(46, 204, 113)
        green_dark = rgb(39, 174, 96)
        green_fluro = rgb(0, 255, 0)

        green2 = rgb(26, 188, 156)
        green2_dark = rgb(22, 160, 133)

        blue = rgb(52, 152, 219)
        blue_dark = rgb(41, 128, 185)
        blue_fluro = rgb(0, 0, 255)

        yellow = rgb(241, 196, 15)

        orange = rgb(243, 156, 18)
        orange_dark = rgb(230, 126, 34)
        orange_red = rgb(211, 84, 0)

        purple = rgb(155, 89, 182)
        purple_dark = rgb(142, 68, 173)

        dark = rgb(52, 73, 94)
        darker = rgb(44, 62, 80)

        gray = rgb(149, 165, 166)
        gray_dark = rgb(127, 140, 141)
        grey = gray
        grey_dark = gray_dark

    class hatches:
        cycle = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..'] + ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.'] + ['/o', '\\|', '-\\', '+o', 'o-', 'O|', 'O.']

    class colors:

        cycle = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c",
                 "#9b59b6", "#34495e", "#1abc9c", "#bdc3c7",
                 "#f1c40f",
                 # alternate
                 "#f368e0", "#5f27cd", "#1dd1a1", "#48dbfb",
                 "#22a6b3", "#6F1E51"
                 ]

        @staticmethod
        def get_cycle():
            return ["#3498db", "#e67e22", "#2ecc71", "#e74c3c",
                    "#9b59b6", "#34495e", "#1abc9c", "#bdc3c7",
                    "#f1c40f",
                    # alternate
                    "#f368e0", "#5f27cd", "#1dd1a1", "#48dbfb",
                    "#22a6b3", "#6F1E51"
                    ]

        red = "#e74c3c"
        red_dark = "#c0392b"

        yellow = "#f1c40f"

        orange = "#f39c12"
        orange_dark = "#e67e22"
        orange_darker = "#d35400"

        green = "#2ecc71"
        green_dark = "#27ae60"

        lighter = "#ecf0f1"
        light = "#bdc3c7"

        grey = "#95a5a6"
        grey_dark = "#7f8c8d"

        black = "#000000"
        white = "#FFFFFF"

        dark = "#34495e"
        darker = "#2c3e50"

        blue = "#3498db"
        blue_dark = "#2980b9"

        purple = "#9b59b6"
        purple_dark = "#8e44ad"
        purple_alt = "#30336b"

        green2 = "#1abc9c"
        green2_dark = "#16a085"

        black = "#000000"

    def equal_axes(self: Axes):
        plt.gca().set_aspect('equal', adjustable='box')

    def xticks(self, rotation: float = None, **kwargs):
        plt.xticks(rotation=rotation, **kwargs)

    @property
    def original_figure(self) -> plt.Figure:
        return self._obj

    def show(self):
        plt.show()

    def colorbar(self, mappable, axes=None, orientation="vertical", **kwargs):
        return plt.colorbar(mappable=mappable, ax=axes, use_gridspec=True, orientation=orientation, fraction=0.05,
                            shrink=0.8, **kwargs)

    def subplots_adjust(self, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        self.original_figure.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace,
                                             hspace=hspace)

class StandardFonts(Enum):
    Roboto = "Roboto"
    Montserrat = "Montserrat"
    Consolas = "Consolas"
    Arial = "Arial"
    TimesNewRoman = "Times New Roman"
    CallingCode = "Calling Code"
    CMUStandard = "CMU Sans Serif"
    CMUBright = "CMU Bright"
    NirmalaUI = "Nirmala UI"

def figure2d(rows=1, cols=1, font_size: int = 16, legend_font_size: int = None, legend_font_scale: float = 1.0,
             sync_axis: Union[bool, Tuple[bool, bool]] = False, center_spines: bool = False,
             font_family: Union[str, StandardFonts] = StandardFonts.NirmalaUI, legend_box=False,
             enhance_axes: bool = True, reload_fonts: bool = False, classic_style: bool = False, legend_alpha:float=0.8) -> Tuple[
    RIFTEnhancedFigure, Union[
        Union[Axes, RIFTEnhancedAxes], List[Union[Axes, RIFTEnhancedAxes]], List[List[Union[Axes, RIFTEnhancedAxes]]]]]:
    if legend_font_size is None:
        legend_font_size = font_size

    if classic_style:
        plt.style.use('classic')
        plt.rc('figure', facecolor='#FFFFFF')
    plt.rcParams['figure.figsize'] = [2, 2]
    plt.rcParams["axes.prop_cycle"] = cycler('color', ["#3498db", "#e67e22", "#2ecc71", "#e74c3c",
                                                       "#9b59b6", "#34495e", "#1abc9c", "#bdc3c7",
                                                       "#f1c40f",
                                                       # alternate
                                                       "#f368e0", "#5f27cd", "#1dd1a1", "#48dbfb",
                                                       "#22a6b3", "#6F1E51"
                                                       ])

    #print(plt.rcParams.keys())

    plt.rcParams["legend.frameon"] = legend_box
    plt.rcParams["legend.framealpha"] = legend_alpha
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.edgecolor"] = "#ecf0f1"
    plt.rcParams["legend.fontsize"] = int(legend_font_size * legend_font_scale)


    primary_color = "#000000"

    plt.rcParams.update({'text.color': primary_color,
                         'axes.labelcolor': primary_color})

    if reload_fonts:
        matplotlib.font_manager._rebuild()

    if font_family is not None:
        font_name = str(font_family if isinstance(font_family, str) else font_family.value)

        from matplotlib import font_manager
        font_manager.findfont(font_name)

        plt.rcParams["font.family"] = font_name

    if isinstance(font_size, int):
        matplotlib.rcParams.update({
            'font.size': font_size
        })

    if isinstance(sync_axis, tuple) or sync_axis:
        sync_x = sync_axis[0] if isinstance(sync_axis, tuple) else sync_axis
        sync_y = sync_axis[1] if isinstance(sync_axis, tuple) else sync_axis
        fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=sync_x, sharey=sync_y)
    else:
        fig, ax = plt.subplots(nrows=rows, ncols=cols)

    fig = fig  # type: plt.Figure
    ax = ax  # type: plt.Axes

    if rows > 1 and cols > 1:
        for r in range(rows):
            for c in range(cols):
                # ax[r, c].patch.set_facecolor('white')
                # ax[r, c].set_facecolor('white')
                ax[r, c].spines['bottom'].set_color(primary_color)
                ax[r, c].spines['top'].set_color(primary_color)
                ax[r, c].spines['right'].set_color(primary_color)
                ax[r, c].spines['left'].set_color(primary_color)
                ax[r, c].tick_params(axis='x', colors=primary_color)
                ax[r, c].tick_params(axis='y', colors=primary_color)

                if center_spines:
                    ax[r, c].spines['left'].set_position('zero')
                    ax[r, c].spines['right'].set_color('none')
                    ax[r, c].spines['bottom'].set_position('zero')
                    ax[r, c].spines['top'].set_color('none')

    elif rows > 1 and cols == 1:
        for c in range(rows):
            # ax[c].patch.set_facecolor('white')
            # ax[c].set_facecolor('white')
            ax[c].spines['bottom'].set_color(primary_color)
            ax[c].spines['top'].set_color(primary_color)
            ax[c].spines['right'].set_color(primary_color)
            ax[c].spines['left'].set_color(primary_color)
            ax[c].tick_params(axis='x', colors=primary_color)
            ax[c].tick_params(axis='y', colors=primary_color)

            if center_spines:
                ax[c].spines['left'].set_position('zero')
                ax[c].spines['right'].set_color('none')
                ax[c].spines['bottom'].set_position('zero')
                ax[c].spines['top'].set_color('none')

    elif rows == 1 and cols > 1:
        for c in range(cols):
            # ax[c].patch.set_facecolor('white')
            # ax[c].set_facecolor('white')
            ax[c].spines['bottom'].set_color(primary_color)
            ax[c].spines['top'].set_color(primary_color)
            ax[c].spines['right'].set_color(primary_color)
            ax[c].spines['left'].set_color(primary_color)
            ax[c].tick_params(axis='x', colors=primary_color)
            ax[c].tick_params(axis='y', colors=primary_color)

            if center_spines:
                ax[c].spines['left'].set_position('zero')
                ax[c].spines['right'].set_color('none')
                ax[c].spines['bottom'].set_position('zero')
                ax[c].spines['top'].set_color('none')


    else:
        # ax.patch.set_facecolor('white')
        # ax.set_facecolor('white')
        ax.spines['bottom'].set_color(primary_color)
        ax.spines['top'].set_color(primary_color)
        ax.spines['right'].set_color(primary_color)
        ax.spines['left'].set_color(primary_color)

        if center_spines:
            ax.spines['left'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_color('none')

        ax.tick_params(axis='x', colors=primary_color)
        ax.tick_params(axis='y', colors=primary_color)

    class EnhancedAxesAccessor:
        def __init__(self, ax):
            self.ax = ax

        def __getitem__(self, item):
            axis = self.ax[item]

            if isinstance(axis, np.ndarray):
                return EnhancedAxesAccessor(axis)

            return RIFTEnhancedAxes(axis)

    return (RIFTEnhancedFigure(fig),
            RIFTEnhancedAxes(ax) if (enhance_axes and rows == 1 and cols == 1) else EnhancedAxesAccessor(ax))
