import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
import re
from pathlib import Path

VLLM_VERSION = "0.13.0"


class BenchmarkMetricLoader:
    @staticmethod
    def _extract_device_folder(summary_csv_path: str) -> str:
        """Extract device folder name from a summary.csv path.

        Expected folder pattern (from benchmark output):
            <DEVICE>_<NUM>_<BACKEND>_<VERSION>
        Examples:
            RNGD_4_furiosa-llm_2026.1.0rc2
            H100_8_vllm_0.13.0
        """
        p = Path(summary_csv_path)
        pat = re.compile(r"^[\w-]+_\d+_(?:furiosa-llm|vllm)_.+")
        for part in p.parts:
            if pat.match(part):
                return part
        # Fallback: keep legacy behavior if we can't find a match.
        parts = summary_csv_path.split("/")
        return parts[-5] if len(parts) >= 5 else "unknown_0_unknown"

    @staticmethod
    def _extract_model_info_line(summary_csv_path: str) -> str:
        """Best-effort extraction of '* Model Info:' header line."""
        try:
            with open(summary_csv_path, "r", encoding="utf-8") as f:
                for _ in range(50):
                    line = f.readline()
                    if not line:
                        break
                    if line.startswith("* Model Info:"):
                        return line.strip()
        except OSError:
            pass
        return ""

    @staticmethod
    def load_offline_benchmark_metric(
        summary_csv_path: str,
    ) -> pd.DataFrame:
        df = pd.read_csv(summary_csv_path, comment="*")
        df.columns = (
            df.columns.str.replace(r"\s*\(.*", "", regex=True)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )

        def col(name: str, default: float = 0.0) -> pd.Series:
            """df_temp에 name이 있으면 numeric Series, 없으면 default로 채운 Series"""
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(default)
            return pd.Series(default, index=df.index, dtype="float64")

        device_folder = BenchmarkMetricLoader._extract_device_folder(summary_csv_path)
        parts = device_folder.split("_")
        device_name = parts[0] if len(parts) > 0 else "Unknown"
        num_devices = parts[1] if len(parts) > 1 else "0"
        runtime = "_".join(parts[2:]) if len(parts) > 2 else ""

        # Prefer including runtime/version info so the report can filter by `--version`.
        if runtime:
            suffix = f" + {runtime}"
        elif not ("RNGD" in device_name or "TARGET" in device_name):
            suffix = f" + vllm-{VLLM_VERSION}"
        else:
            suffix = ""

        if "RNGD" in device_name:
            device_name = device_name.split("-", 1)[0]
        new_df = pd.DataFrame(
            {
                "ISL": col("input_tokens"),
                "OSL": col("output_tokens"),
                "Concurrent": col("concurrent"),
            }
        ).assign(
            **{
                "TPS(Output)": col("output_throughput").round(2),
                "TPS(Total)": col("total_throughput").round(2),
                "TPS/User": (col("output_throughput") / col("concurrent", np.nan)).round(2).fillna(0),
                "TPS/Watt": (col("output_throughput") / col("mean_power", np.nan)).round(2).fillna(0),
                "TTFT(s)": (col("mean_ttft")).round(2),
                "P99_TTFT(s)": (col("p99_ttft")).round(2),
                "TPOT(ms)": (col("mean_tpot")).round(2),
                "P99_TPOT(ms)": (col("p99_tpot")).round(2),
                "E2EL(s)": (col("mean_e2el")).round(2),
                "Power(w)": col("mean_power", 0.0),
                "device": f"{device_name} x {num_devices}{suffix}",
            }
        )
        return new_df


class HyperbolicModel:
    @staticmethod
    def fit_hyperbolic_model(
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float]:
        a_init = np.max(y) * (np.max(x) + np.median(x))
        b_init = max(1.0, float(np.median(x)))

        def hyperbolic_ctp(z, x: float, y: float) -> float:
            return x / (y + z)

        try:
            popt, _ = curve_fit(hyperbolic_ctp, x, y, p0=[a_init, b_init], maxfev=5000)
            a, b = float(popt[0]), float(popt[1])
            if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b >= 0):
                raise Exception
        except (Exception, OptimizeWarning):
            a = np.nan
            b = np.nan
        return (a, b)

    @staticmethod
    def max_conc_from_fit(a: float, b: float, slo: float) -> float:
        if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or slo <= 0:
            return 0.0
        conc_max = a / slo - b
        return max(0.0, float(conc_max))
