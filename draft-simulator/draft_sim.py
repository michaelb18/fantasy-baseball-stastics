from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

positions = pd.read_csv('projections/fangraphs-auction-calculator.csv')

positions['Name'] = positions['Name'].apply(str.strip)

import sys
sys.path.append("../sharpe-ratio")
from sharpe_ratio import read_csvs_batters, read_csvs_pitchers
@dataclass
class Gaussian:
    """Simple Gaussian (normal) distribution used for projections."""

    mean: float
    std: float

    def sample(self, n: int = 1) -> np.ndarray:
        return np.random.normal(self.mean, self.std, size=n)

    def __add__(self, other: Gaussian) -> Gaussian:
        """Sum of independent Gaussians: means add, variances add."""
        return Gaussian(
            mean=self.mean + other.mean,
            std=sqrt(self.std * self.std + other.std * other.std),
        )

    @staticmethod
    def sum(gaussians: List[Gaussian]) -> Gaussian:
        """Sum a list of independent Gaussians (empty => mean=0, std=0)."""
        if not gaussians:
            return Gaussian(mean=0.0, std=0.0)
        out = gaussians[0]
        for g in gaussians[1:]:
            out = out + g
        return out


@dataclass
class BatterProjections:
    hr: Gaussian
    rbi: Gaussian
    runs: Gaussian
    sb: Gaussian
    h: Gaussian
    bb: Gaussian
    ab: Gaussian
    
    @property
    def obp(self) -> float:
        h_samples = self.h.sample(n = 1000)
        bb_samples = self.bb.sample(n = 1000)
        ab_samples = self.ab.sample(n = 1000)
        batter_obp = (h_samples + bb_samples)/ab_samples
        batter_obp = Gaussian(mean=np.mean(batter_obp), std=np.std(batter_obp))

        return batter_obp



@dataclass
class PitcherProjections:
    k: Gaussian
    wins: Gaussian
    svhld: Gaussian
    ip: Gaussian
    bb: Gaussian
    er: Gaussian
    h: Gaussian

    @property
    def whip(self) -> float:
        h_samples = self.h.sample(n = 1000)
        bb_samples = self.bb.sample(n = 1000)
        ip_samples = self.ip.sample(n = 1000)
        pitcher_whip = (h_samples + bb_samples)/ip_samples
        pitcher_whip = Gaussian(mean=np.mean(pitcher_whip), std=np.std(pitcher_whip))

        return pitcher_whip

    @property
    def era(self) -> float:
        er_samples = self.er.sample(n = 1000)
        ip_samples = self.ip.sample(n = 1000)
        pitcher_era = er_samples/ip_samples * 9
        pitcher_era = Gaussian(mean=np.mean(pitcher_era), std=np.std(pitcher_era))

        return pitcher_era


@dataclass
class TeamProjection:
    """Combined Gaussian projections for a team (sum of all rostered players)."""

    # Batting (from batters)
    hr: Gaussian
    rbi: Gaussian
    runs: Gaussian
    sb: Gaussian
    obp: Gaussian
    # Pitching (from pitchers)
    k: Gaussian
    wins: Gaussian
    whip: Gaussian
    era: Gaussian
    svhld: Gaussian


@dataclass
class Player:
    name: str
    team: str
    salary: float
    eligible_positions: List[str] = field(default_factory=list)


@dataclass
class Batter(Player):
    projections: BatterProjections = field(default=None)
    projection_system: str = ""


@dataclass
class Pitcher(Player):
    projections: PitcherProjections = field(default=None)
    projection_system: str = ""


@dataclass
class Team:
    """Represents a fantasy team in an auction league."""

    name: str
    salary_cap: float

    # Position slots, e.g. {"C": 1, "1B": 1, "2B": 1, "SS": 1, "3B": 1, "OF": 5, "UTIL": 1}
    batter_slots: Dict[str, int]
    pitcher_slots: Dict[str, int]

    batters: List[Batter] = field(default_factory=list)
    pitchers: List[Pitcher] = field(default_factory=list)

    max_batters: int = 15
    max_pitchers: int = 9

    @property
    def salary_spent(self) -> float:
        return sum(b.salary for b in self.batters) + sum(p.salary for p in self.pitchers)

    @property
    def salary_remaining(self) -> float:
        return self.salary_cap - self.salary_spent

    def add_batter(self, batter: Batter) -> bool:
        """Try to add a batter to the roster, enforcing cap and roster size."""
        if len(self.batters) >= self.max_batters:
            return False
        if self.salary_spent + batter.salary > self.salary_cap:
            return False
        self.batters.append(batter)
        return True

    def add_pitcher(self, pitcher: Pitcher) -> bool:
        """Try to add a pitcher to the roster, enforcing cap and roster size."""
        if len(self.pitchers) >= self.max_pitchers:
            return False
        if self.salary_spent + pitcher.salary > self.salary_cap:
            return False
        self.pitchers.append(pitcher)
        return True

    def team_projection(self) -> TeamProjection:
        """Sum each stat's Gaussians across all batters and pitchers on this team."""
        batting_hr = Gaussian.sum([b.projections.hr for b in self.batters if b.projections])
        batting_rbi = Gaussian.sum([b.projections.rbi for b in self.batters if b.projections])
        batting_runs = Gaussian.sum([b.projections.runs for b in self.batters if b.projections])
        batting_sb = Gaussian.sum([b.projections.sb for b in self.batters if b.projections])
        batting_h = Gaussian.sum([b.projections.h for b in self.batters if b.projections])
        batting_bb = Gaussian.sum([b.projections.bb for b in self.batters if b.projections])
        batting_ab = Gaussian.sum([b.projections.ab for b in self.batters if b.projections])

        h_samples = batting_h.sample(n = 1000)
        bb_samples = batting_bb.sample(n = 1000)
        ab_samples = batting_ab.sample(n = 1000)
        batting_obp = (h_samples + bb_samples)/ab_samples
        batting_obp = Gaussian(mean=np.mean(batting_obp), std=np.std(batting_obp))

        pitching_k = Gaussian.sum([p.projections.k for p in self.pitchers if p.projections])
        pitching_wins = Gaussian.sum([p.projections.wins for p in self.pitchers if p.projections])
        pitching_er = Gaussian.sum([p.projections.er for p in self.pitchers if p.projections])
        pitching_ip = Gaussian.sum([p.projections.ip for p in self.pitchers if p.projections])
        pitching_bb = Gaussian.sum([p.projections.bb for p in self.pitchers if p.projections])
        pitching_h = Gaussian.sum([p.projections.h for p in self.pitchers if p.projections])
        pitching_svhld = Gaussian.sum([p.projections.svhld for p in self.pitchers if p.projections])

        p_h_samples = pitching_h.sample(n = 1000)
        p_bb_samples = pitching_bb.sample(n = 1000)
        p_ip_samples = pitching_ip.sample(n = 1000)
        p_er_samples = pitching_er.sample(n = 1000)

        pitching_whip = (p_h_samples + p_bb_samples)/p_ip_samples
        pitching_era = 9 * p_er_samples/p_ip_samples

        pitching_whip = Gaussian(mean=np.mean(pitching_whip), std=np.std(pitching_whip))
        pitching_era = Gaussian(mean=np.mean(pitching_era), std=np.std(pitching_era))

        return TeamProjection(
            hr=batting_hr,
            rbi=batting_rbi,
            runs=batting_runs,
            sb=batting_sb,
            obp=batting_obp,
            k=pitching_k,
            wins=pitching_wins,
            whip=pitching_whip,
            era=pitching_era,
            svhld=pitching_svhld,
        )


def _safe_gaussian(mean: float, std: float, fallback_std: float) -> Gaussian:
    """
    Build a Gaussian from mean and per-player std, guarding against
    zero/NaN/inf standard deviations similar in spirit to the sharpe_ratio logic.
    """
    if not np.isfinite(std) or std <= 0.0:
        std = fallback_std
    # Final small epsilon to avoid an exactly-degenerate distribution.
    std = max(std, 1e-6)
    return Gaussian(mean=mean, std=std)


def build_batter_universe(
    salary_by_player: Optional[Dict[str, float]] = None,
    positions_by_player: Optional[Dict[str, List[str]]] = None,
) -> List[Batter]:
    """
    Build Batter instances with Gaussian projections using read_csvs_batters().

    read_csvs_batters() returns a dataframe indexed by player name with
    per-category mean and std across all projection systems; we turn those
    into Gaussian projections for each fantasy batter.
    """
    dfs_sharpe = read_csvs_batters()

    # Fallback std per category: mean of non-zero stds across players.
    def _fallback_std(col: str) -> float:
        std_series = dfs_sharpe[(col, "std")].replace(0.0, np.nan)
        value = float(std_series.mean(skipna=True))
        return max(value, 1e-3)

    fallback_hr_std = _fallback_std("HR")
    fallback_r_std = _fallback_std("R")
    fallback_rbi_std = _fallback_std("RBI")
    fallback_sb_std = _fallback_std("SB")
    fallback_h_std = _fallback_std("H")
    fallback_bb_std = _fallback_std("BB")
    fallback_ab_std = _fallback_std("PA")

    # Map each player to an MLB team using a single projection set (e.g. Steamer).
    team_df = pd.read_csv("projections/hitters/steamer.csv", delimiter="\t")
    team_by_player = dict(zip(team_df["Name"], team_df["Team"]))

    salary_by_player = salary_by_player or {}
    positions_by_player = positions_by_player or {}

    batters: List[Batter] = []

    for _, row in dfs_sharpe.iterrows():
        name = str(row["Name"].values[0])
        team = str(team_by_player.get(name, ""))
        print(row)
        hr_mean = float(row[("HR", "mean")])
        hr_std = float(row[("HR", "std")])
        r_mean = float(row[("R", "mean")])
        r_std = float(row[("R", "std")])
        rbi_mean = float(row[("RBI", "mean")])
        rbi_std = float(row[("RBI", "std")])
        sb_mean = float(row[("SB", "mean")])
        sb_std = float(row[("SB", "std")])
        h_mean = float(row[("H", "mean")])
        h_std = float(row[("H", "std")])
        bb_mean = float(row[("BB", "mean")])
        bb_std = float(row[("BB", "std")])
        ab_mean = float(row[("PA", "mean")])
        ab_std = float(row[("PA", "std")])

        projections = BatterProjections(
            hr=_safe_gaussian(hr_mean, hr_std, fallback_hr_std),
            rbi=_safe_gaussian(rbi_mean, rbi_std, fallback_rbi_std),
            runs=_safe_gaussian(r_mean, r_std, fallback_r_std),
            sb=_safe_gaussian(sb_mean, sb_std, fallback_sb_std),
            h=_safe_gaussian(h_mean, h_std, fallback_h_std),
            bb=_safe_gaussian(bb_mean, bb_std, fallback_bb_std),
            ab=_safe_gaussian(ab_mean, ab_std, fallback_ab_std),
        )
        batter = Batter(
            name=name,
            team=team,
            salary=salary_by_player.get(name, 0.0),
            eligible_positions=positions[positions['Name'] == name]['POS'].values,
            projections=projections,
            projection_system="multi_system",
        )
        batters.append(batter)

    return batters


def build_pitcher_universe(
    salary_by_player: Optional[Dict[str, float]] = None,
    positions_by_player: Optional[Dict[str, List[str]]] = None,
) -> List[Pitcher]:
    """
    Build Pitcher instances with Gaussian projections using read_csvs_pitchers().

    We combine starter and reliever Sharpe-dataframes, preferring the reliever
    row when a player appears in both, and compute Gaussians for W, SO, SVHLD,
    WHIP, and ERA.
    """
    starters_df = read_csvs_pitchers(starters=True)
    relievers_df = read_csvs_pitchers(starters=False)

    combined = pd.concat([starters_df, relievers_df], ignore_index=True)

    # Fallback std per category: mean of non-zero stds across players.
    def _fallback_std(col: str) -> float:
        std_series = combined[(col, "std")].replace(0.0, np.nan)
        value = float(std_series.mean(skipna=True))
        return max(value, 1e-3)

    fallback_w_std = _fallback_std("W")
    fallback_k_std = _fallback_std("SO")
    fallback_svhld_std = _fallback_std("SVHLD")
    fallback_whip_std = _fallback_std("WHIP")
    fallback_era_std = _fallback_std("ERA")

    # Map each player to an MLB team using a single projection set (e.g. Steamer).
    team_df = pd.read_csv("projections/pitchers/steamer.csv", delimiter="\t")
    team_by_player = dict(zip(team_df["Name"], team_df["Team"]))

    salary_by_player = salary_by_player or {}
    positions_by_player = positions_by_player or {}

    # Prefer reliever rows when a player appears in both, otherwise starter rows.
    rows_by_name: Dict[str, pd.Series] = {}
    for _, row in starters_df.iterrows():
        name = str(row["Name"].values[0])
        rows_by_name.setdefault(name, row)
    for _, row in relievers_df.iterrows():
        name = str(row["Name"].values[0])
        rows_by_name[name] = row

    pitchers: List[Pitcher] = []

    for name, row in rows_by_name.items():
        team = str(team_by_player.get(name, ""))

        w_mean = float(row[("W", "mean")])
        w_std = float(row[("W", "std")])
        k_mean = float(row[("SO", "mean")])
        k_std = float(row[("SO", "std")])

        if ("SVHLD", "mean") in row.index:
            svhld_mean = float(row[("SVHLD", "mean")])
            svhld_std = float(row[("SVHLD", "std")])
        else:
            svhld_mean = 0.0
            svhld_std = 0.0

        whip_mean = float(row[("WHIP", "mean")])
        whip_std = float(row[("WHIP", "std")])
        era_mean = float(row[("ERA", "mean")])
        era_std = float(row[("ERA", "std")])
        ip_mean = float(row[("IP", "mean")])
        ip_std = float(row[("IP", "std")])
        h_mean = float(row[("H", "mean")])
        h_std = float(row[("H", "std")])
        bb_mean = float(row[("BB", "mean")])
        bb_std = float(row[("BB", "std")])
        er_mean = float(row[("ER", "mean")])
        er_std = float(row[("ER", "std")])

        projections = PitcherProjections(
            k=_safe_gaussian(k_mean, k_std, fallback_k_std),
            wins=_safe_gaussian(w_mean, w_std, fallback_w_std),
            svhld=_safe_gaussian(svhld_mean, svhld_std, fallback_svhld_std),
            ip=_safe_gaussian(ip_mean, ip_std, _fallback_std("IP")),
            bb=_safe_gaussian(bb_mean, bb_std, _fallback_std("BB")),
            er=_safe_gaussian(er_mean, er_std, _fallback_std("ER")),
            h=_safe_gaussian(h_mean, h_std, _fallback_std("H")),
        )

        pitcher = Pitcher(
            name=name,
            team=team,
            salary=salary_by_player.get(name, 0.0),
            eligible_positions=[],
            projections=projections,
            projection_system="multi_system",
        )
        pitchers.append(pitcher)

    return pitchers

if __name__ == "__main__":
    # Simple manual test: build universes and print counts.
    batters = build_batter_universe()
    pitchers = build_pitcher_universe()
    print(f"Loaded {len(batters)} batters and {len(pitchers)} pitchers.")
