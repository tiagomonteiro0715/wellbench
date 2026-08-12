from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Physical bounds used for clipping synthetic outputs AND cleaning real data
# ---------------------------------------------------------------------------
PHYSICAL_BOUNDS = {
    "GR":     (0,       200),
    "DT":     (30,      180),
    "RHOB":   (1.2,     2.9),
    "RT":     (0.01,    10_000),
    "HP":     (500,     15_000),
    "OB":     (2_000,   40_000),
    "DT_NCT": (30,      180),
    "PPP":    (0,       30_000),
}

SENTINEL_VALUES = {-999, -999.25, -999.0}


# ---------------------------------------------------------------------------
# Cleaning utility (works on real AND synthetic DataFrames)
# ---------------------------------------------------------------------------
def clean_well_data(
    df: pd.DataFrame,
    outlier_std: float = 5,
    verbose: bool = True,
    label: str = "",
    physical_bounds: dict | None = None,
) -> pd.DataFrame:
    """Clean a single well DataFrame (synthetic OR real).

    1. Drop SPHI column if present.
    2. Replace sentinel values (-999, -999.25) with NaN.
    3. Replace values outside physical bounds with NaN.
    4. Replace extreme outliers (> *outlier_std* sigma) with NaN.
    5. Drop rows where ALL log columns are NaN.
    """
    bounds = physical_bounds or PHYSICAL_BOUNDS
    df = df.copy()
    n_before = len(df)

    if "SPHI" in df.columns:
        df.drop(columns=["SPHI"], inplace=True)

    log_cols = [c for c in df.columns if c != "DEPTH"]

    # Sentinel → NaN
    for col in log_cols:
        df.loc[df[col].isin(SENTINEL_VALUES), col] = np.nan

    # Physical bounds → NaN
    for col in log_cols:
        if col in bounds:
            lo, hi = bounds[col]
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # Outlier removal
    for col in log_cols:
        clean_vals = df[col].dropna()
        if len(clean_vals) < 10:
            continue
        mu, sig = clean_vals.mean(), clean_vals.std()
        if sig == 0:
            continue
        df.loc[(df[col] - mu).abs() > outlier_std * sig, col] = np.nan

    df.dropna(subset=log_cols, how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if verbose:
        removed = n_before - len(df)
        pct = 100 * removed / n_before if n_before else 0
        print(
            f"  [{label}]  {n_before} -> {len(df)} rows  "
            f"(removed {removed}, {pct:.1f}%)"
        )
    return df


class SyntheticWellLogGenerator:
    """Generates synthetic well-log data from physics-based petrophysical models.

    Given a region parameter dictionary (with depth range and rock/fluid properties),
    produces DT, GR, RHOB, RT (and optionally HP, OB, DT_NCT, PPP for regions
    with pore-pressure parameters).
    """

    def __init__(self, params: dict):
        self.p = params

    # ------------------------------------------------------------------
    # Core physics: porosity from Athy's compaction law
    # ------------------------------------------------------------------
    def make_porosity(self, depth: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        trend = self.p["phi0"] * np.exp(-self.p["compaction_c"] * depth)
        layers = self.p["phi_layer_amp"] * np.sin(depth / self.p["phi_layer_period"])
        noise = rng.normal(0, self.p["phi_noise_std"], len(depth))
        return np.clip(trend + layers + noise, 0.02, 0.50)

    # ------------------------------------------------------------------
    # Porosity -> log conversions
    # ------------------------------------------------------------------
    def phi_to_dt(self, phi: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        dt = phi * self.p["dt_fluid"] + (1 - phi) * self.p["dt_matrix"]
        dt += rng.normal(0, self.p["dt_noise_std"], len(phi))
        return np.clip(dt, *PHYSICAL_BOUNDS["DT"])

    def phi_to_rhob(self, phi: np.ndarray, depth: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        lith_var = 0.1 * np.sin(depth / 120)
        rho_matrix_var = self.p["rho_matrix"] + lith_var
        rhob = phi * self.p["rho_fluid"] + (1 - phi) * rho_matrix_var
        rhob += rng.normal(0, self.p["rhob_noise_std"], len(phi))
        return np.clip(rhob, *PHYSICAL_BOUNDS["RHOB"])

    def phi_to_res(self, phi: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # Guard phi against underflow: small phi + large archie_m → Rt explodes
        phi_safe = np.clip(phi, 0.05, 0.50)
        rt = (self.p["archie_a"] * self.p["Rw"]
              / (phi_safe ** self.p["archie_m"] * self.p["Sw"] ** self.p["archie_n"]))
        log_rt = np.log10(rt) + rng.normal(0, self.p["res_log_noise_std"], len(phi))
        return np.clip(10 ** log_rt, self.p["res_clip_min"], self.p["res_clip_max"])

    def phi_to_gr(self, phi: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        phi_norm = np.clip((phi - 0.02) / (0.50 - 0.02), 0, 1)
        gr = self.p["gr_shale"] + phi_norm * (self.p["gr_clean"] - self.p["gr_shale"])
        gr += rng.normal(0, self.p["gr_noise_std"], len(phi))
        return np.clip(gr, *PHYSICAL_BOUNDS["GR"])

    # ------------------------------------------------------------------
    # Derived pore-pressure curves
    # ------------------------------------------------------------------
    def calc_hp(self, depth: np.ndarray) -> np.ndarray:
        return self.p["psi_per_ft_per_gcc"] * self.p["hp_fluid_density"] * depth

    def calc_ob(self, depth: np.ndarray, rhob: np.ndarray) -> np.ndarray:
        c = self.p["psi_per_ft_per_gcc"]
        ob = np.zeros_like(depth, dtype=float)
        ob[0] = self.p["ob_shallow_rhob"] * c * depth[0]
        rho_avg = (rhob[:-1] + rhob[1:]) / 2
        dz = np.gradient(depth[:-1])
        ob[1:] = ob[0] + np.cumsum(rho_avg * c * dz)
        return ob

    def calc_dt_nct(self, depth: np.ndarray, dt: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=float)
        dt = np.asarray(dt, dtype=float)

        trend = (self.p["nct_dt0"]
                 + (self.p["nct_dt_surface"] - self.p["nct_dt0"])
                 * np.exp(-self.p["nct_k"] * depth))

        win = int(max(3, min(self.p["nct_smooth_window"], len(dt))))
        dt_smooth = uniform_filter1d(dt, size=win, mode="nearest")

        w = float(np.clip(self.p["nct_weight_smooth"], 0.0, 1.0))
        dt_nct = w * dt_smooth + (1.0 - w) * trend
        return np.clip(dt_nct, *PHYSICAL_BOUNDS["DT_NCT"])

    def calc_ppp(self, dt: np.ndarray, dt_nct: np.ndarray,
                 ob: np.ndarray, hydro: np.ndarray) -> np.ndarray:
        n = self.p["eaton_n"]
        return ob - (ob - hydro) * (dt / dt_nct) ** n

    # ------------------------------------------------------------------
    # Build a single synthetic dataset
    # ------------------------------------------------------------------
    def generate(self, seed: int, depth: np.ndarray | None = None) -> pd.DataFrame:
        if depth is None:
            start, stop = self.p["depth_range"]
            depth = np.arange(start, stop, self.p["depth_step"])

        phi = self.make_porosity(depth, seed)
        dt = self.phi_to_dt(phi, seed + 10)
        rhob = self.phi_to_rhob(phi, depth, seed + 20)
        res = self.phi_to_res(phi, seed + 30)
        gr = self.phi_to_gr(phi, seed + 40)

        data = {
            "DEPTH": depth,
            "GR": gr,
            "DT": dt,
            "RHOB": rhob,
            "RT": res,
        }

        if self.p.get("has_pore_pressure", False):
            hp = self.calc_hp(depth)
            ob = self.calc_ob(depth, rhob)
            dt_nct = self.calc_dt_nct(depth, dt)

            # Clip intermediates BEFORE Eaton so PPP can't blow up
            dt = np.clip(dt, *PHYSICAL_BOUNDS["DT"])
            dt_nct = np.clip(dt_nct, *PHYSICAL_BOUNDS["DT_NCT"])
            ob = np.clip(ob, *PHYSICAL_BOUNDS["OB"])
            hp = np.clip(hp, *PHYSICAL_BOUNDS["HP"])

            ppp = self.calc_ppp(dt, dt_nct, ob, hp)

            data["DT"] = dt  # update with clipped version
            data.update({
                "HP": hp,
                "OB": ob,
                "DT_NCT": dt_nct,
                "PPP": ppp,
            })

        # Final physical bounds on everything
        for col, (lo, hi) in PHYSICAL_BOUNDS.items():
            if col in data:
                data[col] = np.clip(data[col], lo, hi)

        return pd.DataFrame(data)
