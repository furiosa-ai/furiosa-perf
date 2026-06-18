from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit


class BenchmarkMetricLoader:
    @staticmethod
    def _extract_device_and_version(summary_csv_path: str | Path) -> tuple[str, str, str, str]:
        """Extract device metadata from a summary.csv path.

        The result folder follows the benchmark output pattern
        ``<DEVICE>_<NUM>_<BACKEND>[_<VERSION>]`` (e.g.
        ``RNGD_4_furiosa-llm_2026.1.0rc2`` or ``H100-80GB_8_vllm_0.13.0``). The
        trailing ``<VERSION>`` segment is optional: ``furiosa-perf run`` writes
        version-less folders such as ``RNGD_4_furiosa-llm``, in which case the
        version is returned as an empty string.

        Args:
            summary_csv_path (str): Path to a ``summary.csv`` benchmark result file.

        Returns:
            tuple[str, str, str, str]: The ``(device, num, backend, version)`` parsed
            from the result folder name.
        """
        p = Path(summary_csv_path)
        parts = p.parents[3].name.split("_")
        device, num, backend = parts[0], parts[1], parts[2]
        version = "_".join(parts[3:])  # "" when no version segment is present
        return device, num, backend, version

    @staticmethod
    def load_offline_benchmark_metric(
        summary_csv_path: str | Path,
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
                "Median_TTFT(s)": (col("median_ttft")).round(2),
                "TPOT(ms)": (col("mean_tpot")).round(2),
                "P99_TPOT(ms)": (col("p99_tpot")).round(2),
                "Median_TPOT(ms)": (col("median_tpot")).round(2),
                "E2EL(s)": (col("mean_e2el")).round(2),
                "Power(w)": col("mean_power", 0.0),
                "device": f"{device}x{num}+{backend}_{version}",
                "hardware": device,
                "backend": backend,
                "version": version,
            }
        )
        return new_df


def interp_conc_from_tps_user(
    conc: np.ndarray,
    tps_user: np.ndarray,
    slo: float,
) -> float:
    """Estimate the maximum concurrency whose per-user throughput still meets an SLO.

    The feasible region is ``{Concurrent : TPS/User >= slo}`` and we return its right
    edge (the largest concurrency that still meets the SLO), obtained by
    piecewise-linear interpolation of the actual (Concurrent, TPS/User) samples.
    This is robust to non-monotonic curves: TPS/User often *rises* before it falls
    (e.g. Concurrent=4 yielding a higher TPS/User than Concurrent=1), so:

    * if no measured point reaches ``slo`` (the SLO is above the achievable peak),
      nothing is feasible and the result is ``0``;
    * we take the *largest* SLO crossing, so a rising left-hand segment can never be
      extrapolated into a spurious feasible point.

    This tracks the data far more faithfully than fitting a single global hyperbolic
    curve, which tends to be inaccurate between samples.

    Args:
        conc (np.ndarray): Measured concurrency (number of concurrent users) for each
            sample.
        tps_user (np.ndarray): Measured per-user output throughput (TPS/User) for each
            sample, aligned element-wise with ``conc``.
        slo (float): Target minimum per-user throughput (the SLO threshold).

    Returns:
        float: The largest concurrency that still satisfies the SLO, or ``0.0`` when no
        sample reaches ``slo`` or the inputs are empty/invalid.
    """
    conc = np.asarray(conc, dtype="float64")
    tps = np.asarray(tps_user, dtype="float64")

    mask = np.isfinite(conc) & np.isfinite(tps)
    conc, tps = conc[mask], tps[mask]
    if conc.size == 0 or slo <= 0:
        return 0.0

    order = np.argsort(conc)
    conc, tps = conc[order], tps[order]

    # The SLO exceeds the best per-user throughput ever measured -> nothing is feasible.
    if not np.any(tps >= slo):
        return 0.0

    if conc.size == 1:
        return float(conc[0])  # single point, and it is feasible (checked above)

    # Largest concurrency on a segment that crosses the SLO level: the right edge of
    # the feasible region.
    best: float | None = None
    for i in range(conc.size - 1):
        y0, y1 = tps[i], tps[i + 1]
        if y0 == y1:
            continue
        if (y0 - slo) * (y1 - slo) <= 0:  # slo lies within this segment
            t = (slo - y0) / (y1 - y0)
            xc = float(conc[i] + t * (conc[i + 1] - conc[i]))
            best = xc if best is None else max(best, xc)

    # Feasible all the way to the highest measured concurrency (SLO looser than every
    # sample): extrapolate along the decreasing tail.
    if tps[-1] >= slo:
        y0, y1 = tps[-2], tps[-1]
        x0, x1 = conc[-2], conc[-1]
        if y1 < y0:
            t = (slo - y0) / (y1 - y0)
            xc = float(x0 + t * (x1 - x0))
            best = xc if best is None else max(best, xc)
        else:
            tail = float(conc[-1])
            best = tail if best is None else max(best, tail)

    return max(0.0, best) if best is not None else 0.0


class HyperbolicModel:
    @staticmethod
    def fit_hyperbolic_model(
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float]:
        a_init = np.max(y) * (np.max(x) + np.median(x))
        b_init = max(1.0, float(np.median(x)))

        def hyperbolic_ctp(z: np.ndarray, x: float, y: float) -> np.ndarray:
            return np.asarray(x / (y + z), dtype="float64")

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
