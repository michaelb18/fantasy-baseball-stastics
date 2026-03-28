import configparser
import copy
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]

# Make sure we can import the draft simulator and sharpe-ratio helpers
sys.path.append(str(BASE_DIR / "draft-simulator"))
sys.path.append(str(BASE_DIR / "sharpe-ratio"))

#from sharpe_ratio import read_csvs_batters as build_batter_universe, read_csvs_pitchers as build_pitcher_universe
from draft_sim import (  # type: ignore  # noqa: E402
    Batter,
    Pitcher,
    Team,
    TeamProjection,
    build_batter_universe,
    build_pitcher_universe,
)

TEAMS_INI_PATH = BASE_DIR / "draft-simulator" / "teams.ini"


def load_team_configs(path: pathlib.Path = TEAMS_INI_PATH) -> Dict[str, Dict]:
    config = configparser.ConfigParser()
    config.read(path)

    def parse_slots(raw: str) -> Dict[str, int]:
        slots: Dict[str, int] = {}
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                continue
            pos, count = part.split(":", 1)
            try:
                slots[pos.strip()] = int(count.strip())
            except ValueError:
                continue
        return slots

    def parse_keepers(raw: str) -> List[Tuple[str, float]]:
        """Parse 'Name:salary, Name:salary, ...' into [(name, salary), ...]."""
        keepers: List[Tuple[str, float]] = []
        for part in raw.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            name, salary_str = part.split(":", 1)
            name = name.strip()
            try:
                keepers.append((name, float(salary_str.strip())))
            except ValueError:
                continue
        return keepers

    teams: Dict[str, Dict] = {}
    for section in config.sections():
        salary_cap = config.getfloat(section, "salary_cap", fallback=260.0)
        batter_slots_raw = config.get(section, "batter_slots", fallback="UTIL:15")
        pitcher_slots_raw = config.get(section, "pitcher_slots", fallback="P:9")
        keepers_raw = config.get(section, "keepers", fallback="")

        teams[section] = {
            "salary_cap": salary_cap,
            "batter_slots": parse_slots(batter_slots_raw),
            "pitcher_slots": parse_slots(pitcher_slots_raw),
            "keepers": parse_keepers(keepers_raw),
        }

    return teams


def initialize_league() -> None:
    if "teams" in st.session_state:
        return

    team_cfgs = load_team_configs()

    batters = build_batter_universe()
    pitchers = build_pitcher_universe()

    teams: Dict[str, Team] = {}
    for name, cfg in team_cfgs.items():
        teams[name] = Team(
            name=name,
            salary_cap=cfg["salary_cap"],
            batter_slots=cfg["batter_slots"],
            pitcher_slots=cfg["pitcher_slots"],
        )

    st.session_state["teams"] = teams
    st.session_state["batters_by_name"] = {b.name: b for b in batters}
    st.session_state["pitchers_by_name"] = {p.name: p for p in pitchers}

    # Auto-draft keepers from teams.ini (name:salary per team).
    batters_by_name = st.session_state["batters_by_name"]
    pitchers_by_name = st.session_state["pitchers_by_name"]
    for team_name, cfg in team_cfgs.items():
        team = teams[team_name]
        for keeper_name, keeper_salary in cfg.get("keepers", []):
            keeper_name = keeper_name.strip()
            if not keeper_name:
                continue
            if keeper_name in batters_by_name:
                template = batters_by_name[keeper_name]
                player = copy.deepcopy(template)
                player.salary = float(keeper_salary)
                if team.add_batter(player):
                    del batters_by_name[keeper_name]
            elif keeper_name in pitchers_by_name:
                template = pitchers_by_name[keeper_name]
                player = copy.deepcopy(template)
                player.salary = float(keeper_salary)
                if team.add_pitcher(player):
                    del pitchers_by_name[keeper_name]

    st.session_state["initialized"] = True


def get_teams() -> Dict[str, Team]:
    initialize_league()
    return st.session_state["teams"]


def draft_player_to_team(player_name: str, salary: float, team_name: str) -> str:
    teams: Dict[str, Team] = get_teams()
    team = teams.get(team_name)
    if team is None:
        return f"Unknown team: {team_name}"

    player_name = player_name.strip()
    if not player_name:
        return "Player name cannot be empty."

    batters_by_name: Dict[str, Batter] = st.session_state["batters_by_name"]
    pitchers_by_name: Dict[str, Pitcher] = st.session_state["pitchers_by_name"]

    template_player = None
    is_batter = False

    if player_name in batters_by_name:
        template_player = batters_by_name[player_name]
        is_batter = True
    elif player_name in pitchers_by_name:
        template_player = pitchers_by_name[player_name]
        is_batter = False
    else:
        return f"Player '{player_name}' not found in projections."

    # Use a copy so salary and team assignment are independent of global pool.
    player = copy.deepcopy(template_player)
    player.salary = float(salary)

    if is_batter:
        added = team.add_batter(player)
    else:
        added = team.add_pitcher(player)

    if not added:
        return f"Could not add {player_name} to {team_name} (roster full or salary cap exceeded)."

    # Remove from draftable pool so they can't be drafted again.
    if is_batter:
        del batters_by_name[player_name]
    else:
        del pitchers_by_name[player_name]

    return f"Drafted {player_name} to {team_name} for ${salary:.0f}."


def _batters_dataframe(batters: List[Batter]) -> pd.DataFrame:
    rows = []
    for b in batters:
        proj = b.projections
        rows.append(
            {
                "Name": b.name,
                "MLB Team": b.team,
                "Pos": ", ".join(b.eligible_positions),
                "Salary": b.salary,
                "HR μ": proj.hr.mean if proj else np.nan,
                "HR σ": proj.hr.std if proj else np.nan,
                "R μ": proj.runs.mean if proj else np.nan,
                "R σ": proj.runs.std if proj else np.nan,
                "RBI μ": proj.rbi.mean if proj else np.nan,
                "RBI σ": proj.rbi.std if proj else np.nan,
                "SB μ": proj.sb.mean if proj else np.nan,
                "SB σ": proj.sb.std if proj else np.nan,
                "OBP μ": proj.obp.mean if proj else np.nan,
                "OBP σ": proj.obp.std if proj else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _pitchers_dataframe(pitchers: List[Pitcher]) -> pd.DataFrame:
    rows = []
    for p in pitchers:
        proj = p.projections
        rows.append(
            {
                "Name": p.name,
                "MLB Team": p.team,
                "Pos": ", ".join(p.eligible_positions) or "P",
                "Salary": p.salary,
                "K μ": proj.k.mean if proj else np.nan,
                "K σ": proj.k.std if proj else np.nan,
                "W μ": proj.wins.mean if proj else np.nan,
                "W σ": proj.wins.std if proj else np.nan,
                "SV+HLD μ": proj.svhld.mean if proj else np.nan,
                "SV+HLD σ": proj.svhld.std if proj else np.nan,
                "ERA μ": proj.era.mean if proj else np.nan,
                "ERA σ": proj.era.std if proj else np.nan,
                "WHIP μ": proj.whip.mean if proj else np.nan,
                "WHIP σ": proj.whip.std if proj else np.nan,
            }
        )
    return pd.DataFrame(rows)


# Categories for points ranking (1st=16 pts, 2nd=15, ...). Lower-is-better for ERA/WHIP.
BATTER_CATEGORIES = ["hr", "rbi", "runs", "sb", "obp"]
PITCHER_CATEGORIES = ["k", "wins", "svhld", "whip", "era"]
ALL_CATEGORIES = BATTER_CATEGORIES + PITCHER_CATEGORIES
LOWER_IS_BETTER = {"whip", "era"}


def _projection_value(proj: TeamProjection, category: str, scenario: str) -> float:
    """Get scalar value for one category: 'best' (mean+3*std), 'mean', or 'worst' (mean-3*std)."""
    g = getattr(proj, category)
    if scenario == "best":
        if category.lower() in LOWER_IS_BETTER:
            return g.mean - 3 * g.std
        else:
            return g.mean + 3 * g.std
    elif scenario == "worst":
        if category.lower() in LOWER_IS_BETTER:
            return g.mean + 3 * g.std
        else:
            return g.mean - 3 * g.std
    elif scenario == "sharpe":
        return g.mean / g.std
    elif scenario = "mean":
        return g.mean
    raise ValueError("Scenario must be one of {'best', 'worst', 'mean', 'sharpe'}")


def _compute_rankings(
    teams: Dict[str, Team], scenario: str
) -> pd.DataFrame:
    """Rank teams by points: 10 categories, 1st=16 pts down to 16th=1 pt."""
    team_names = sorted(teams.keys())
    n = len(team_names)
    points_per_place = list(range(n, 0, -1))  # 16,15,...,1 if n=16

    # Per-team values for each category
    values: Dict[str, Dict[str, float]] = {t: {} for t in team_names}
    for name, team in teams.items():
        p = team.team_projection()
        for cat in ALL_CATEGORIES:
            values[name][cat] = _projection_value(p, cat, scenario)

    # Rank in each category and assign points
    ranks: Dict[str, Dict[str, int]] = {t: {} for t in team_names}
    for cat in ALL_CATEGORIES:
        reverse = cat not in LOWER_IS_BETTER
        sorted_teams = sorted(
            team_names, key=lambda t: values[t][cat], reverse=reverse
        )
        for rank_idx, t in enumerate(sorted_teams):
            ranks[t][cat] = points_per_place[rank_idx]

    # Build table: Team, category point columns, Total. Column labels for display.
    cat_display = ["HR", "RBI", "R", "SB", "OBP", "K", "W", "SV+HLD", "WHIP", "ERA"]
    rows = []
    for t in team_names:
        row = {"Team": t}
        total = 0
        for cat, disp in zip(ALL_CATEGORIES, cat_display):
            row[disp] = ranks[t][cat]
            total += ranks[t][cat]
        row["Total"] = total
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("Total", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank
    return df


def _compute_dollars_per_player_rankings(teams: Dict[str, Team]) -> pd.DataFrame:
    """
    Rank teams by dollars per remaining roster spot.

    dollars_per_player = salary_remaining / (batters_spots_left + pitcher_spots_left)

    This is shown as a standalone ranking (not added into stat-category points).
    """
    team_names = sorted(teams.keys())
    rows = []

    for name in team_names:
        team = teams[name]
        batters_left = max(0, team.max_batters - len(team.batters))
        pitchers_left = max(0, team.max_pitchers - len(team.pitchers))
        spots_left = batters_left + pitchers_left
        dollars_left = float(team.salary_remaining)

        dollars_per_player = dollars_left / spots_left if spots_left > 0 else None

        rows.append(
            {
                "Team": name,
                "Batters left": batters_left,
                "Pitchers left": pitchers_left,
                "Spots left": spots_left,
                "$ left": round(dollars_left, 2),
                "$/player": round(dollars_per_player, 2) if dollars_per_player is not None else None,
            }
        )

    df = pd.DataFrame(rows)

    # Sort by $/player descending; teams with 0 spots left go last.
    df["_has_spots"] = df["Spots left"] > 0
    df = df.sort_values(
        by=["_has_spots", "$/player"],
        ascending=[False, False],
    ).drop(columns=["_has_spots"]).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank
    return df


def _team_projection_table(proj: TeamProjection) -> pd.DataFrame:
    """Table with rows Best case (mean+3*std), Mean, Worst case (mean-3*std)."""
    scenarios = [
        ("Best case (μ+3σ)", "best"),
        ("Mean (μ)", "mean"),
        ("Worst case (μ−3σ)", "worst"),
        ("Sharpe (Risk Adjusted)", "sharpe")
    ]
    categories_display = [
        ("HR", "hr"),
        ("RBI", "rbi"),
        ("R", "runs"),
        ("SB", "sb"),
        ("OBP", "obp"),
        ("K", "k"),
        ("W", "wins"),
        ("SV+HLD", "svhld"),
        ("WHIP", "whip"),
        ("ERA", "era"),
    ]
    rows = []
    for label, scenario in scenarios:
        row = {"Scenario": label}
        for disp, cat in categories_display:
            row[disp] = _projection_value(proj, cat, scenario)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Fantasy Baseball Auction Draft", layout="wide")
    st.title("Fantasy Baseball Auction Draft Simulator")

    teams = get_teams()
    team_names = sorted(teams.keys())

    st.sidebar.header("Navigation")
    page_options = ["Home"] + team_names
    page_choice = st.sidebar.selectbox(
        "Page",
        page_options,
        format_func=lambda x: "Home (rankings)" if x == "Home" else x,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("Teams and roster settings are defined in `teams.ini`.")

    if page_choice == "Home":
        # Home page: best, average, and worst case rankings with points
        st.header("Dollars per player remaining")
        st.markdown(
            "This ranks teams by **salary remaining ÷ remaining roster spots**. "
            "Example: if you have 10 total players left to draft and $160 left, your $/player is 16. "
            "This is **not** a scoring category and does not affect the stat-category points tables."
        )
        st.dataframe(
            _compute_dollars_per_player_rankings(teams),
            use_container_width=True,
            height=400,
        )

        st.markdown("---")

        st.header("League rankings by projected points")
        st.markdown(
            "Teams are ranked in each of 10 categories (HR, RBI, R, SB, OBP, K, W, SV+HLD, WHIP, ERA). "
            "1st place = 16 pts, 2nd = 15 pts, … 16th = 1 pt. Total = sum of category points."
        )
        for scenario_label, scenario_key in [
            ("Best case (μ+3σ)", "best"),
            ("Average case (μ)", "mean"),
            ("Worst case (μ−3σ)", "worst"),
        ]:
            st.subheader(scenario_label)
            df = _compute_rankings(teams, scenario_key)
            st.dataframe(df, use_container_width=True, height=400)

    else:
        # Team page: selected team roster + projection scenarios + draft form
        selected_team_name = page_choice
        selected_team = teams[selected_team_name]

        st.subheader(f"Team: {selected_team.name}")

        # Team projection: best / mean / worst
        proj = selected_team.team_projection()
        st.markdown("#### Projected totals (best / mean / worst case)")
        st.dataframe(
            _team_projection_table(proj),
            use_container_width=True,
        )

        col_info, col_draft = st.columns([2, 1])

        with col_info:
            st.metric("Salary cap", f"${selected_team.salary_cap:.0f}")
            st.metric("Salary spent", f"${selected_team.salary_spent:.0f}")
            st.metric("Salary remaining", f"${selected_team.salary_remaining:.0f}")

            st.markdown("#### Batters")
            batters_df = _batters_dataframe(selected_team.batters)
            st.dataframe(batters_df, use_container_width=True, height=300)

            st.markdown("#### Pitchers")
            pitchers_df = _pitchers_dataframe(selected_team.pitchers)
            st.dataframe(pitchers_df, use_container_width=True, height=300)

        with col_draft:
            st.subheader("Draft Player")
            with st.form("draft_form"):
                player_name = st.text_input("Player name")
                salary = st.number_input(
                    "Salary", min_value=1.0, max_value=1000.0, value=10.0, step=1.0
                )
                team_choice = st.selectbox(
                    "Team",
                    team_names,
                    index=team_names.index(selected_team_name),
                )
                submitted = st.form_submit_button("Draft")

            if submitted:
                message = draft_player_to_team(player_name, salary, team_choice)
                if "Drafted" in message:
                    st.success(message)
                else:
                    st.error(message)


if __name__ == "__main__":
    main()

