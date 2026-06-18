import colorsys
from dataclasses import dataclass

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


# Base hue (degrees on the HSL color wheel) per hardware family. Each device gets a
# distinct shade *generated* from its family hue, so a family with an arbitrary number
# of devices still receives unique, automatically-assigned colors.
FAMILY_HUE = {
    "RNGD": 5,  # red
    "RTX": 212,  # blue
    "H100": 140,  # green
    "H200": 168,  # teal-green
    "A100": 43,  # amber
    "TARGET": 0,  # neutral grey (saturation forced to 0 below)
}
DEFAULT_HUE = 285  # fallback for unrecognized families


def _device_family(device_name: str) -> str:
    """Classify a hardware name into a color family.

    Args:
        device_name (str): The hardware portion of a device label
            (e.g. ``"RTX-PRO-6000"``).

    Returns:
        str: The family key (``"RNGD"``, ``"RTX"``, ``"H100"``, ``"H200"``,
        ``"A100"``, ``"TARGET"``) or ``"OTHER"`` when the name matches no known family.
    """
    name = device_name.upper()
    if "RNGD" in name:
        return "RNGD"
    if "RTX" in name:
        return "RTX"
    if "H100" in name:
        return "H100"
    if "H200" in name:
        return "H200"
    if "A100" in name:
        return "A100"
    if "TARGET" in name:
        return "TARGET"
    return "OTHER"


def _shade(hue_deg: float, idx: int, count: int, saturation: float) -> str:
    """Generate a distinct color within a family.

    Members of a family share a base hue but are spread over lightness (plus a small
    hue jitter) so they stay separable however many there are.

    Args:
        hue_deg (float): Base hue of the family, in degrees on the HSL wheel.
        idx (int): Index of this member within the family (0-based).
        count (int): Total number of members in the family.
        saturation (float): HSL saturation in ``[0, 1]`` (0 yields a grey ramp).

    Returns:
        str: The color as a ``#RRGGBB`` hex string.
    """
    if count <= 1:
        lightness, hue = 0.60, hue_deg
    else:
        lightness = 0.72 - 0.34 * (idx / (count - 1))  # light -> dark
        hue = hue_deg + (idx - (count - 1) / 2.0) * 10.0  # small spread
    r, g, b = colorsys.hls_to_rgb((hue % 360) / 360.0, lightness, saturation)
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


def build_device_color_map(devices: list[str]) -> dict[str, str]:
    """Map every device to a theme color, grouped by hardware family.

    Families are colored by hue (RNGD = red, RTX = blue, H100 = green, ...). Colors
    are generated rather than read from a fixed list, so adding more devices to a
    family keeps assigning unique shades automatically instead of running out of
    entries.

    Args:
        devices (list[str]): Device labels of the form
            ``"<hardware>x<num>+<backend>_<version>"``.

    Returns:
        dict[str, str]: A mapping from each device label to its ``#RRGGBB`` hex color.
    """
    families: dict[str, list[str]] = {}
    for device in devices:
        hardware = device.split("+")[0].split("x")[0]
        families.setdefault(_device_family(hardware), []).append(device)

    colors: dict[str, str] = {}
    for family, members in families.items():
        members = sorted(members)
        hue = FAMILY_HUE.get(family, DEFAULT_HUE)
        saturation = 0.0 if family == "TARGET" else 0.62
        for idx, device in enumerate(members):
            colors[device] = _shade(hue, idx, len(members), saturation)
    return colors


SERVER_POWER_CONSUMPTION = {
    "H100-80GB": 10200,
    "A100-80GB": 7600,
    "RTX-PRO-6000": 6600,
    "H200": 5000,
}


# ----------------------------------------------------------------------------
# Landing-page summary (RNGD vs RTX Pro 6000 performance-ratio bar charts)
# ----------------------------------------------------------------------------

# The summary compares the NPU (RNGD, latest furiosa-llm) against the GPU
# (RTX Pro 6000, latest vLLM) for a single ISL/OSL scenario, broken down by
# concurrency. Tune these to retarget the comparison.
SUMMARY_SCENARIO_ISL_OSL = (1024, 1024)
SUMMARY_NPU_FAMILY = "RNGD"
SUMMARY_NPU_BACKEND = "furiosa-llm"
SUMMARY_GPU_FAMILY = "RTX"
SUMMARY_GPU_BACKEND = "vllm"

# Horizontal dashed reference line drawn on every summary chart (0.9 == 90%).
SUMMARY_REFERENCE_RATIO = 0.9

# One bar chart per entry. ``ratio`` is "npu/gpu" or "gpu/npu" so that a taller
# bar always means "RNGD is better"; ``better`` is the direction of the raw
# metric and decides how repeated configs at one concurrency are reduced
# (best throughput = max, best latency = min).
SUMMARY_METRICS = [
    {"key": "tps_user", "label": "TPS / User", "column": "TPS/User", "better": "high", "ratio": "npu/gpu"},
    {
        "key": "tps_watt",
        "label": "TPS / Watt (Power Efficiency)",
        "column": "TPS/Watt",
        "better": "high",
        "ratio": "npu/gpu",
    },
    {"key": "median_ttft", "label": "Median TTFT", "column": "Median_TTFT(s)", "better": "low", "ratio": "gpu/npu"},
    {"key": "median_tpot", "label": "Median TPOT", "column": "Median_TPOT(ms)", "better": "low", "ratio": "gpu/npu"},
]

# Qualitative palette cycled across models in the summary charts.
MODEL_COLOR_PALETTE = [
    "#FF5A5F",
    "#3D9BFF",
    "#FFB400",
    "#22C55E",
    "#A855F7",
    "#14B8A6",
    "#F97316",
    "#EC4899",
    "#64748B",
    "#84CC16",
]


def build_model_color_map(models: list[str]) -> dict[str, str]:
    """Assign each model a stable color by cycling :data:`MODEL_COLOR_PALETTE`.

    Args:
        models (list[str]): Model names to color (order defines color assignment).

    Returns:
        dict[str, str]: Mapping from model name to a ``#RRGGBB`` hex color.
    """
    return {model: MODEL_COLOR_PALETTE[i % len(MODEL_COLOR_PALETTE)] for i, model in enumerate(models)}
