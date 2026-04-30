import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from pathlib import Path


class BenchmarkMetricLoader:
    @staticmethod
    def _extract_device_and_version(summary_csv_path: str) -> tuple[str, str]:
        """Extract device folder name from a summary.csv path.

        Expected folder pattern (from benchmark output):
            <DEVICE>_<BACKEND>_<VERSION>_<NUM>
        Examples:
            RNGD_furiosa-llm_2026.1.0rc2_4
            H100_vllm_0.13.0_8
        """
        p = Path(summary_csv_path)
        a, b, c, d = p.parents[3].name.split("_")
        return a, b, c, d

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
            """If the DataFrame contains 'name', return a numeric Series.
            Otherwise, return a Series filled with the default value."""
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(default)
            return pd.Series(default, index=df.index, dtype="float64")

        device, num, backend, version = BenchmarkMetricLoader._extract_device_and_version(summary_csv_path)

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
                "device": f"{device}x{num}+{backend}_{version}",
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
            popt, _ = curve_fit(
                hyperbolic_ctp,
                x,
                y,
                p0=[a_init, b_init],
                maxfev=5000,
                sigma=np.ones_like(y) * 0.01,
                absolute_sigma=True,
            )
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
