from dataclasses import dataclass

import numpy as np

TABLE_COLUMNS = ["ISL", "OSL", "Concurrent", "TPS/User", "TPS(Output)", "TTFT(s)", "TPOT(ms)", "Power(w)"]
INIT_SLO = 20.0


# Table Chart Settings
@dataclass
class TableChartSettings:
    col_w: int = 33
    col_ratio: float = 9.5
    header_h: int = 37
    cell_h: int = 35
    header_font_size: int = 17
    cell_font_size: int = 16
    font_family: str = "Favorit-medium, system-ui"
    even_row_color: str = "#000000"
    odd_row_color: str = "#121212"


@dataclass
class LineChartSettings:
    font_family: str = "Favorit-medium, system-ui"
    panel_w: int = 260
    font_color: str = "#eeeeee"
    font_size: int = 14


@dataclass
class InteractiveChartSettings:
    font_family: str = "Favorit-medium, system-ui"
    font_color: str = "#eeeeee"
    font_size: int = 14
    column_widths_left: float = 0.75
    column_height_upper: float = 0.55


DEVICE_COLOR_MAP = {
    "RNGD": ["#5A0000", "#FF0000", "#E21500", "#FF3D00", "#FF6D00", "#FF5252", "#FF8787"],
    "TARGET": ["#9B9B9B", "#A78581", "#B36E67", "#CA4234"],
    "A100-80GB": ["#FFFA82", "#70E697", "#76D6FF"],
    "A100-PCIe": [],
    "H100-80GB": ["#A7C7E7", "#A8D5BA", "#ffffff", "#ffffff"],
    "H100-PCIe": [],
    "RTX-PRO-6000": ["#F7E1A0", "#9AD0EC", "#F6C28B"],
}


def get_device_color(info: str, RNGD_IDX: list[tuple[str, int]]) -> str:
    hardware_info, backend_info = info.split("+")
    device_name, used_device_num = hardware_info.split("x")
    color = "#000000"
    if "RNGD" not in device_name and "TARGET" not in device_name:
        device_color_list = DEVICE_COLOR_MAP[device_name]
        color = device_color_list[int(np.log2(int(used_device_num)))]
    elif "RNGD" in device_name:
        device_color_list = DEVICE_COLOR_MAP["RNGD"]
        color = device_color_list[RNGD_IDX[info]]
    elif "TARGET" in device_name:
        device_color_list = DEVICE_COLOR_MAP["TARGET"]
        if "99.9" in device_name:
            color = device_color_list[0]
        elif "80" in device_name:
            color = device_color_list[1]
        elif "60" in device_name:
            color = device_color_list[2]
        elif "50" in device_name:
            color = device_color_list[3]

    return color


SERVER_POWER_CONSUMPTION = {
    "H100-80GB": 10200,
    "A100-80GB": 7600,
    "RTX-PRO-6000": 6600,
    "H200": 5000,
}
