import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from furiosa_perf.reporting.schemas import interp_conc_from_tps_user
from furiosa_perf.reporting.theme import (
    INIT_SLO,
    SERVER_POWER_CONSUMPTION,
    InteractiveChartSettings,
    LineChartSettings,
    TableChartSettings,
    build_device_color_map,
)


def plot_table_chart(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    table_chart_theme = TableChartSettings()

    view_df = df.loc[:, columns].copy()
    table_width_per_column = [
        12 + table_chart_theme.col_w + table_chart_theme.col_ratio * max(len(str(c)), 7) for c in columns
    ]
    table_height = table_chart_theme.header_h + len(view_df) * table_chart_theme.cell_h

    row_colors = [
        table_chart_theme.odd_row_color if i % 2 == 0 else table_chart_theme.even_row_color for i in range(len(view_df))
    ] * len(columns)
    cell_colors = [row_colors] * len(columns)

    table_chart_fig = go.Figure(
        data=[
            go.Table(
                columnwidth=table_width_per_column,
                header={
                    "values": [c.replace(" ", "\u00A0") for c in columns],
                    "align": "center",
                    "height": table_chart_theme.header_h,
                    "fill_color": table_chart_theme.even_row_color,
                    "font": {
                        "family": table_chart_theme.font_family,
                        "size": table_chart_theme.header_font_size,
                        "color": "#ffffff",
                    },
                    "line": {"width": 0},
                    "line_color": "rgba(255,255,255,0.0)",
                },
                cells={
                    "values": [view_df[c].tolist() for c in columns],
                    "align": "center",
                    "height": table_chart_theme.cell_h,
                    "fill_color": cell_colors,
                    "font": {
                        "family": table_chart_theme.font_family,
                        "size": table_chart_theme.cell_font_size,
                        "color": "#ffffff",
                    },
                    "line": {"width": 0},
                    "line_color": "rgba(255,255,255,0.0)",
                },
                meta={"role": "rngd_table"},
            )
        ]
    )

    table_chart_fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=table_height,
        width=sum(table_width_per_column) + 80,
        autosize=False,
    )

    for i in range(0, len(view_df)):
        y_px = round(table_chart_theme.header_h + i * table_chart_theme.cell_h) + 0.5
        y = 1.0 - (y_px / table_height)
        table_chart_fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=y,
            y1=y,
            xref="paper",
            yref="paper",
            line={"color": "rgba(255,255,255,0.35)", "width": 1.2},
            layer="above",
        )

    return table_chart_fig


def plot_line_chart(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
) -> go.Figure:
    line_chart_theme = LineChartSettings()

    device_colors = build_device_color_map(sorted(df["device"].unique()))
    line_chart_fig = px.line(
        df,
        x=x_axis,
        y=y_axis,
        color="device",
        color_discrete_map=device_colors,
        markers=True,
    )

    for i, tr in enumerate(line_chart_fig.data):
        tr.legendgroup = f"g{i}"

    line_chart_fig.for_each_trace(lambda tr: tr.update(legendgroup=tr.name))
    line_chart_fig.for_each_trace(lambda tr: tr.update(line={"dash": "dot"}) if "TARGET" in (tr.name or "") else None)
    line_chart_fig.update_xaxes(
        type="linear",
        range=[0, df[x_axis].max() * 1.1],
        gridcolor="rgba(255,255,255,0.12)",
        ticks="outside",
        ticklen=5,
    )
    line_chart_fig.update_yaxes(
        range=[0, df[y_axis].max() * 1.1],
        gridcolor="rgba(255,255,255,0.12)",
        ticks="outside",
        ticklen=5,
    )
    line_chart_fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        margin={"l": 40, "r": line_chart_theme.panel_w + 30, "t": 30, "b": 30},
        legend={
            "traceorder": "grouped",
            "tracegroupgap": 12,
            "font": {
                "family": line_chart_theme.font_family,
                "size": line_chart_theme.font_size,
                "color": line_chart_theme.font_color,
            },
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
        },
        shapes=[
            {
                "type": "rect",
                "xref": "paper",
                "yref": "paper",
                "xsizemode": "pixel",
                "xanchor": 1.0,
                "x0": 0,
                "x1": line_chart_theme.panel_w + 30,
                "y0": 0,
                "y1": 1.0,
                "fillcolor": "rgba(20,20,20,0.65)",
                "line": {"color": "rgba(255,255,255,0.15)", "width": 1},
                "layer": "below",
            }
        ],
        autosize=True,
    )
    return line_chart_fig


def plot_interactive_user_chart(
    df: pd.DataFrame,
) -> go.Figure:
    interactive_chart_theme = InteractiveChartSettings()
    device_colors = build_device_color_map(sorted(df["device"].unique()))
    entire_chart = make_subplots(
        rows=2,
        cols=2,
        column_widths=[interactive_chart_theme.column_widths_left, 1.0 - interactive_chart_theme.column_widths_left],
        row_heights=[interactive_chart_theme.column_height_upper, 1.0 - interactive_chart_theme.column_height_upper],
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy", "colspan": 1}, None]],
        vertical_spacing=0.15,
        horizontal_spacing=0.0,  # (FIX) 0.02 -> 0
    )

    line_chart_fig = plot_line_chart(df, "TPS/User", "TPS(Output)")

    for trace in sorted(line_chart_fig.data, key=lambda t: (t.name or "").lower()):
        trace.legendgroup = trace.name
        entire_chart.add_trace(trace, row=1, col=1)

    entire_chart.update_xaxes(
        type="linear",
        title_text="TPS/User",
        row=1,
        col=1,
        range=[0, df["TPS/User"].max() * 1.1],
        gridcolor="rgba(255,255,255,0.12)",
        ticks="outside",
        ticklen=5,
    )
    entire_chart.update_yaxes(
        title_text="TPS(Output)",
        row=1,
        col=1,
        range=[0, df["TPS(Output)"].max() * 1.1],
        gridcolor="rgba(255,255,255,0.12)",
        ticks="outside",
        ticklen=5,
    )

    entire_chart.add_shape(
        type="line",
        yref="paper",
        x0=INIT_SLO,
        x1=INIT_SLO,
        y0=0,
        y1=df["TPS(Output)"].max(),
        line={"width": 3, "dash": "dash", "color": "white"},
        name="slo_line",
        row=1,
        col=1,
    )

    num_devices = 0
    devices = []
    curve_conc = []
    curve_tps_user = []
    init_supported_list = []
    for device, group in sorted(df.groupby("device")):
        g = group.sort_values("Concurrent")
        conc = g["Concurrent"].to_numpy()
        tps_user = g["TPS/User"].to_numpy()
        init_supported = interp_conc_from_tps_user(conc, tps_user, INIT_SLO)
        devices.append(device)
        curve_conc.append(conc.tolist())
        curve_tps_user.append(tps_user.tolist())
        init_supported_list.append(init_supported)
        entire_chart.add_trace(
            go.Bar(
                x=[device],
                y=[init_supported],
                name=device,
                legendgroup=device,
                showlegend=False,
                marker_color=device_colors[device],
            ),
            row=2,
            col=1,
        )
        num_devices += 1
    entire_chart.update_yaxes(
        title_text="Users",
        row=2,
        col=1,
        range=[0, max(init_supported_list) * 1.05],
        autorange=True,
        fixedrange=False,
        gridcolor="rgba(255,255,255,0.12)",
        ticks="outside",
        ticklen=5,
    )
    entire_chart.update_xaxes(
        title_text="", row=2, col=1, range=[-1.5, (num_devices - 1) + 1.5], ticks="outside", ticklen=5
    )
    entire_chart.update_xaxes(visible=False, row=1, col=2)
    entire_chart.update_yaxes(visible=False, row=1, col=2)

    entire_chart.update_layout(
        title={"text": "", "subtitle": {"text": ""}},
        plot_bgcolor="black",
        paper_bgcolor="black",
        font={
            "family": interactive_chart_theme.font_family,
            "color": interactive_chart_theme.font_color,
            "size": interactive_chart_theme.font_size,
        },
        margin={"l": 40, "r": 40, "t": 30, "b": 30},
        meta={"labels": devices, "curveConc": curve_conc, "curveTpsU": curve_tps_user},
        legend={
            "orientation": "v",
            "x": interactive_chart_theme.column_widths_left + 0.01,
            "y": 1.03,
            "xanchor": "left",
            "yanchor": "top",
            # Native legend background sizes itself to the entries (and scrolls when
            # taller than the plot), so device names never spill outside the panel as
            # the device count grows.
            "bgcolor": "rgba(20,20,20,0.65)",
            "bordercolor": "rgba(255,255,255,0.15)",
            "borderwidth": 1,
            "itemsizing": "constant",
            "font": {
                "family": interactive_chart_theme.font_family,
                "color": interactive_chart_theme.font_color,
                "size": interactive_chart_theme.font_size,
            },
            "groupclick": "togglegroup",
        },
        bargap=0.3,
        height=900,
    )

    return entire_chart


def plot_rack_performance_chart(df: pd.DataFrame) -> go.Figure:
    rack_performance_chart = go.Figure()
    device_metadata = []
    device_colors = build_device_color_map(sorted(df["device"].unique()))
    for device, group in sorted(df.groupby("device")):
        x = group["Concurrent"].to_numpy()
        y = group["TPS/User"].to_numpy()
        max_conc = interp_conc_from_tps_user(x, y, INIT_SLO)
        device_name, info = device.split("x")
        num_device, _ = info.split("+")
        server_power = (
            3000 if ("TARGET" in device_name or "RNGD" in device_name) else SERVER_POWER_CONSUMPTION[device_name]
        )
        users_per_rack = max_conc * (8 / int(num_device))

        rack_power_list: list[float] = [0]
        users_rack_list: list[float] = [0]

        n = 1
        while n * server_power <= 36000:
            rack_power_list.append(n * server_power)
            users_rack_list.append(n * users_per_rack)
            n += 1

        users_rack_list.append(users_rack_list[-1])
        rack_power_list.append(36000)
        dash = "solid" if "TARGET" not in device_name else "dot"
        rack_performance_chart.add_trace(
            go.Scatter(
                x=rack_power_list,
                y=users_rack_list,
                mode="lines+markers",
                name=device_name,
                line_shape="hv",
                line=dict[str, int | str](width=3, dash=dash, color=device_colors[device]),
            )
        )

        device_metadata.append(
            {
                "device_name": device_name,
                "tps_per_dev": users_rack_list,
                "num_device": int(num_device),
                "initial_power": server_power,
                "color": device_colors[device],
            }
        )

    rack_performance_chart.update_xaxes(
        gridcolor="rgba(255,255,255,0.15)", ticks="outside", ticklen=5, rangemode="tozero"
    )
    rack_performance_chart.update_yaxes(
        gridcolor="rgba(255,255,255,0.15)",
        ticks="outside",
        ticklen=5,
        linecolor="rgba(255,255,255,0.15)",
        rangemode="tozero",
    )

    rack_performance_chart.update_layout(
        xaxis_title="Power Capacity of Rack (kw)",
        yaxis_title="Users per Rack",
        font={"family": "Favorit-medium, system-ui", "color": "#ffffff", "size": 14},
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        width=750,
        height=620,
        legend={
            "orientation": "h",
            # 길어지면 여러 줄로 wrap 되도록 "항목 폭" 제한
            "entrywidthmode": "fraction",
            "entrywidth": 0.22,  # 0.18~0.30 사이에서 조절 (작을수록 더 잘 줄바꿈)
            "x": 0.0,
            "xanchor": "left",
            "y": 1.08,
            "yanchor": "bottom",  # 상단 바깥으로
            "traceorder": "normal",  # grouped 필요 없으면 normal 권장(가로에서는 체감 적음)
            "font": {"family": "Favorit-medium, system-ui", "color": "#eeeeee", "size": 14},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
        },
        margin={"r": 40, "l": 40, "t": 30, "b": 30},
        meta={"device_metadata": device_metadata, "max_rack_kw": 36000},
    )
    return rack_performance_chart


def plot_ttft_or_tpot_chart(df: pd.DataFrame, target_metric: str) -> go.Figure:
    entire_chart = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        vertical_spacing=0.15,
    )
    metric = "s" if target_metric == "ttft" else "ms"
    mean_line_chart = plot_line_chart(df, "Concurrent", f"{target_metric.upper()}({metric})")
    p99_line_chart = plot_line_chart(df, "Concurrent", f"P99_{target_metric.upper()}({metric})")

    for trace in sorted(mean_line_chart.data, key=lambda t: (t.name or "").lower()):
        trace.legendgroup = trace.name
        entire_chart.add_trace(trace, row=1, col=1)

    for trace in sorted(p99_line_chart.data, key=lambda t: (t.name or "").lower()):
        trace.showlegend = False
        entire_chart.add_trace(trace, row=1, col=2)

    entire_chart.update_yaxes(
        title_text=f"Mean {target_metric.upper()}({metric})",
        row=1,
        col=1,
        linecolor="rgba(0,0,0,0.0)",
        gridcolor="rgba(255,255,255,0.12)",
        ticks="outside",
        rangemode="tozero",
        ticklen=5,
    )
    entire_chart.update_xaxes(
        title_text="Concurrent",
        row=1,
        col=1,
        ticks="outside",
        ticklen=5,
        rangemode="tozero",
    )
    entire_chart.update_yaxes(
        title_text=f"P99 {target_metric.upper()}({metric})",
        row=1,
        col=2,
        linecolor="rgba(0,0,0,0.0)",
        gridcolor="rgba(255,255,255,0.12)",
        ticks="outside",
        rangemode="tozero",
        ticklen=5,
    )
    entire_chart.update_xaxes(
        title_text="Concurrent",
        row=1,
        col=2,
        ticks="outside",
        ticklen=5,
        rangemode="tozero",
    )

    entire_chart.update_layout(
        title={"text": "", "subtitle": {"text": ""}},
        plot_bgcolor="black",
        paper_bgcolor="black",
        font={
            "family": "Favorit-medium, system-ui",
            "color": "#ffffff",
            "size": 14,
        },
        margin={"l": 40, "r": 40, "t": 30, "b": 30},
        legend={
            "orientation": "h",
            # 길어지면 여러 줄로 wrap 되도록 "항목 폭" 제한
            "entrywidthmode": "fraction",
            "entrywidth": 0.22,  # 0.18~0.30 사이에서 조절 (작을수록 더 잘 줄바꿈)
            "x": 0.0,
            "xanchor": "left",
            "y": 1.08,
            "yanchor": "bottom",  # 상단 바깥으로
            "traceorder": "normal",  # grouped 필요 없으면 normal 권장(가로에서는 체감 적음)
            "font": {"family": "Favorit-medium, system-ui", "color": "#eeeeee", "size": 14},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
        },
        bargap=0.3,
    )
    return entire_chart
