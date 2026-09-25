"""
EuroLeague game dashboard.

Pick a season and game, then view the scoreboard, game flow, team & player
stats, betting-relevant game facts and the full play-by-play.

    streamlit run app.py
"""
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DB_PATH = Path(__file__).resolve().parent / "data" / "processed" / "euroleague.db"

# Categorical slots 1 & 2 of the reference palette: home = blue, away = orange
HOME_COLOR = "#2a78d6"
AWAY_COLOR = "#eb6834"

SEASON_LABELS = {"E2024": "2024-25", "E2025": "2025-26", "E2026": "2026-27"}

SHOT_TYPES = {"2FGM", "2FGA", "3FGM", "3FGA", "FTM", "FTA"}
MADE_SHOT_TYPES = {"2FGM", "3FGM", "FTM"}
SHOT_KIND = {"2FGM": ("2PT", 2), "2FGA": ("2PT", 2), "3FGM": ("3PT", 3),
             "3FGA": ("3PT", 3), "FTM": ("FT", 1), "FTA": ("FT", 1)}

PLAY_CATEGORIES = {
    "Shots": {"2FGM", "2FGA", "3FGM", "3FGA", "FTM", "FTA"},
    "Rebounds": {"O", "D"},
    "Assists / Steals / Blocks": {"AS", "ST", "FV", "AG"},
    "Turnovers": {"TO"},
    "Fouls": {"CM", "RV", "OF", "CMU", "CMT", "CMT1", "CMD", "CMTI", "CMU_DI", "CMU_FL", "C", "B"},
    "Substitutions": {"IN", "OUT"},
    "Timeouts / Other": {"TOUT", "TOUT_TV", "CCH", "JB", "BP", "EP", "EG"},
}

st.set_page_config(page_title="EuroLeague Game Dashboard", page_icon="🏀", layout="wide")


# ---------------------------------------------------------------- data access
@st.cache_data
def load_games() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        games = pd.read_sql("SELECT * FROM games ORDER BY season, game_code", conn)
    return games


@st.cache_data
def load_plays(season: str, game_code: int) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        plays = pd.read_sql(
            "SELECT * FROM plays WHERE season = ? AND game_code = ? "
            "ORDER BY quarter_num, play_number",
            conn,
            params=(season, int(game_code)),
        )
    return add_game_clock(plays)


def add_game_clock(plays: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `period` (Q1..Q4, OT1, OT2..) and `t` (game seconds elapsed).
    The API lumps every overtime into one block, so OT periods are split on
    their Begin Period events; clock-less events inherit the previous time.
    """
    plays = plays.copy()
    is_ot = plays["quarter_num"] >= 5
    ot_idx = ((is_ot & (plays["play_type"] == "BP")).cumsum() - 1).clip(lower=0)
    plays["period"] = [f"Q{q}" if q <= 4 else f"OT{o + 1}" for q, o in zip(plays["quarter_num"], ot_idx)]

    offset = plays["quarter_num"].clip(upper=5).sub(1).mul(600) + ot_idx.where(is_ot, 0) * 300
    length = pd.Series(600, index=plays.index).where(~is_ot, 300)
    remaining = plays["seconds_remaining_in_quarter"]
    plays["t"] = (offset + length - remaining).ffill().fillna(0)
    return plays


def game_end(plays: pd.DataFrame) -> float:
    n_ot = plays.loc[plays["period"].str.startswith("OT"), "period"].nunique()
    return 2400.0 + 300 * n_ot


# ---------------------------------------------------------------- stats
def quarter_scores(plays: pd.DataFrame, game: pd.Series) -> pd.DataFrame:
    by_q = plays.groupby("period", sort=False)[["points_scored_a", "points_scored_b"]].sum()
    table = pd.DataFrame(
        {q: [int(r.points_scored_a), int(r.points_scored_b)] for q, r in by_q.iterrows()},
        index=[game.team_a_name, game.team_b_name],
    )
    table["Final"] = [int(game.final_score_a), int(game.final_score_b)]
    return table


def team_stats(plays: pd.DataFrame, team_code: str) -> dict:
    t = plays[plays["team_code"] == team_code]
    n = t["play_type"].value_counts()
    c = lambda k: int(n.get(k, 0))

    fg2m, fg2a = c("2FGM"), c("2FGM") + c("2FGA")
    fg3m, fg3a = c("3FGM"), c("3FGM") + c("3FGA")
    ftm, fta = c("FTM"), c("FTM") + c("FTA")
    pct = lambda m, a: f"{m / a:.1%}" if a else "-"
    return {
        "Points": 2 * fg2m + 3 * fg3m + ftm,
        "Field goals": f"{fg2m + fg3m}/{fg2a + fg3a} ({pct(fg2m + fg3m, fg2a + fg3a)})",
        "2-pointers": f"{fg2m}/{fg2a} ({pct(fg2m, fg2a)})",
        "3-pointers": f"{fg3m}/{fg3a} ({pct(fg3m, fg3a)})",
        "Free throws": f"{ftm}/{fta} ({pct(ftm, fta)})",
        "Rebounds (off / def)": f"{c('O') + c('D')} ({c('O')} / {c('D')})",
        "Assists": c("AS"),
        "Steals": c("ST"),
        "Turnovers": c("TO"),
        "Blocks": c("FV"),
        "Fouls committed": c("CM") + c("OF") + c("CMU") + c("CMT") + c("CMD"),
        "Timeouts": c("TOUT"),
    }


def player_box(plays: pd.DataFrame, team_code: str) -> pd.DataFrame:
    t = plays[(plays["team_code"] == team_code) & plays["player_id"].notna()]
    if t.empty:
        return pd.DataFrame()
    counts = pd.crosstab([t["dorsal"], t["player_name"]], t["play_type"])
    c = lambda k: counts[k] if k in counts else 0

    box = pd.DataFrame(index=counts.index)
    box["PTS"] = 2 * c("2FGM") + 3 * c("3FGM") + c("FTM")
    box["2PM"], box["2PA"] = c("2FGM"), c("2FGM") + c("2FGA")
    box["3PM"], box["3PA"] = c("3FGM"), c("3FGM") + c("3FGA")
    box["FTM"], box["FTA"] = c("FTM"), c("FTM") + c("FTA")
    box["OREB"], box["DREB"] = c("O"), c("D")
    box["REB"] = box["OREB"] + box["DREB"]
    box["AST"], box["STL"], box["TOV"] = c("AS"), c("ST"), c("TO")
    box["BLK"] = c("FV")
    box["PF"] = c("CM") + c("OF") + c("CMU") + c("CMT") + c("CMD")
    box["FD"] = c("RV")

    box = box.reset_index().rename(columns={"dorsal": "#", "player_name": "Player"})
    return box.sort_values(["PTS", "REB"], ascending=False).reset_index(drop=True)


def scoring_events(plays: pd.DataFrame) -> pd.DataFrame:
    return plays[plays["is_scoring_play"] == 1].copy()


def flow_metrics(plays: pd.DataFrame, game: pd.Series) -> dict:
    s = scoring_events(plays)
    lead = s["score_lead_a"].tolist()

    lead_changes, ties, prev_sign = 0, 0, 0
    for v in lead:
        sign = (v > 0) - (v < 0)
        if sign == 0:
            ties += 1
        elif prev_sign and sign != prev_sign:
            lead_changes += 1
        if sign:
            prev_sign = sign

    # Longest unanswered scoring run per team
    best = {"A": 0, "B": 0}
    run_team, run_pts = None, 0
    for _, r in s.iterrows():
        team = "A" if r.points_scored_a > 0 else "B"
        pts = r.points_scored_a if team == "A" else r.points_scored_b
        run_pts = run_pts + pts if team == run_team else pts
        run_team = team
        best[team] = max(best[team], run_pts)

    # Seconds spent leading (lead holds from one scoring play until the next)
    lead_time = {"A": 0.0, "B": 0.0}
    if not s.empty:
        times = s["t"].tolist() + [game_end(plays)]
        for i, v in enumerate(lead):
            dur = max(0, times[i + 1] - times[i])
            if v > 0:
                lead_time["A"] += dur
            elif v < 0:
                lead_time["B"] += dur

    return {
        "lead_changes": lead_changes,
        "ties": ties,
        "max_lead_a": max([0] + lead),
        "max_lead_b": -min([0] + lead),
        "run_a": int(best["A"]),
        "run_b": int(best["B"]),
        "lead_time_a": lead_time["A"],
        "lead_time_b": lead_time["B"],
    }


def game_facts(plays: pd.DataFrame, game: pd.Series) -> pd.DataFrame:
    s = scoring_events(plays)
    name = {game.team_a_code: game.team_a_name, game.team_b_code: game.team_b_name}
    facts = []

    def team_of(row):
        return name.get(row.team_code, row.team_code)

    # First shot attempts (same definition as src/first_shot_analysis.py: FGs and FTs)
    shots = plays[plays["play_type"].isin(SHOT_TYPES)]

    def describe_shot(row, with_team=True):
        kind, value = SHOT_KIND[row.play_type]
        outcome = f"✅ Made (+{value})" if row.play_type in MADE_SHOT_TYPES else "❌ Missed"
        who = f"{team_of(row)} – {row.player_name}" if with_team else row.player_name
        return f"{outcome} · {kind} · {who} ({row.marker_time} {row.period})"

    for label, pos in (("First", 0), ("Last", -1)):
        if not shots.empty:
            facts.append((f"{label} shot of game", describe_shot(shots.iloc[pos])))
        for code, team in ((game.team_a_code, game.team_a_name), (game.team_b_code, game.team_b_name)):
            team_shots = shots[shots["team_code"] == code]
            if not team_shots.empty:
                facts.append((f"{label} shot · {team}", describe_shot(team_shots.iloc[pos], with_team=False)))

    if not s.empty:
        first = s.iloc[0]
        facts.append(("First points", f"{team_of(first)} – {first.player_name} ({first.play_info})"))
        fg = s[s["play_type"].isin(["2FGM", "3FGM"])]
        if not fg.empty:
            f = fg.iloc[0]
            facts.append(("First field goal", f"{team_of(f)} – {f.player_name} ({'3PT' if f.play_type == '3FGM' else '2PT'})"))
        last = s.iloc[-1]
        facts.append(("Last points", f"{team_of(last)} – {last.player_name} ({last.play_info}, {last.marker_time} {last.period})"))

        for target in (10, 20, 30):
            hit = s[(s["current_score_a"] >= target) | (s["current_score_b"] >= target)]
            if not hit.empty:
                h = hit.iloc[0]
                winner = game.team_a_name if h.current_score_a >= target else game.team_b_name
                facts.append((f"Race to {target}", f"{winner} ({int(h.current_score_a)}–{int(h.current_score_b)}, {h.marker_time} {h.period})"))

    half = plays[plays["quarter_num"] <= 2][["points_scored_a", "points_scored_b"]].sum()
    facts.append(("Half-time score", f"{int(half.points_scored_a)}–{int(half.points_scored_b)} (total {int(half.sum())})"))
    total = int(game.final_score_a + game.final_score_b)
    facts.append(("Final total points", str(total)))
    margin = int(game.final_score_a - game.final_score_b)
    facts.append(("Winning margin", f"{game.team_a_name if margin > 0 else game.team_b_name} by {abs(margin)}"))
    facts.append(("Overtime", "Yes" if game.actual_quarter > 4 else "No"))
    return pd.DataFrame(facts, columns=["Fact", "Value"])


# ---------------------------------------------------------------- charts
def _period_bounds(end: float) -> list:
    """(start, stop, label) for every period up to `end`."""
    bounds = [(i * 600, (i + 1) * 600, f"Q{i + 1}") for i in range(4)]
    for i, start in enumerate(range(2400, int(end), 300)):
        bounds.append((start, start + 300, f"OT{i + 1}"))
    return bounds


def _layout(fig: go.Figure, end: float, height: int = 340):
    bounds = _period_bounds(end)
    ticks = [(a + b) / 2 for a, b, _ in bounds]
    labels = [lbl for _, _, lbl in bounds]
    for _, stop, _ in bounds[:-1]:
        fig.add_vline(x=stop, line_width=1, line_dash="dot", line_color="rgba(128,128,128,0.45)")
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(tickvals=ticks, ticktext=labels, showgrid=False, range=[0, end])
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)


def _timeline(plays: pd.DataFrame) -> tuple:
    """Game timeline from 0 to the final buzzer, with hover labels."""
    end = game_end(plays)
    tl = plays[["t", "period", "marker_time", "current_score_a", "current_score_b", "score_lead_a"]]
    first = {"t": 0, "period": "Q1", "marker_time": "10:00", "current_score_a": 0, "current_score_b": 0, "score_lead_a": 0}
    last = tl.iloc[-1].to_dict() | {"t": end, "marker_time": "00:00"}
    tl = pd.concat([pd.DataFrame([first]), tl, pd.DataFrame([last])], ignore_index=True)
    return tl, end, tl["period"] + " " + tl["marker_time"].fillna("")


def score_chart(plays: pd.DataFrame, game: pd.Series) -> go.Figure:
    tl, end, custom = _timeline(plays)

    fig = go.Figure()
    for col, team, color in (("current_score_a", game.team_a_name, HOME_COLOR),
                             ("current_score_b", game.team_b_name, AWAY_COLOR)):
        fig.add_trace(go.Scatter(
            x=tl["t"], y=tl[col], name=team, mode="lines",
            line=dict(color=color, width=2, shape="hv"), customdata=custom,
            hovertemplate="%{customdata}: %{y}<extra>" + team + "</extra>",
        ))
        fig.add_annotation(x=end, y=tl[col].iloc[-1], text=f" {int(tl[col].iloc[-1])}",
                           showarrow=False, xanchor="left", font=dict(size=12))
    _layout(fig, end)
    fig.update_layout(margin=dict(r=40))
    return fig


def lead_chart(plays: pd.DataFrame, game: pd.Series) -> go.Figure:
    tl, end, custom = _timeline(plays)
    lead = tl["score_lead_a"]
    leader = lead.map(lambda v: f"{game.team_a_code} +{v}" if v > 0 else f"{game.team_b_code} +{-v}" if v < 0 else "Tied")

    fig = go.Figure()
    # Fills only - one invisible trace below carries the hover so it reads once
    fig.add_trace(go.Scatter(
        x=tl["t"], y=lead.clip(lower=0), name=f"{game.team_a_name} leads",
        mode="lines", line=dict(color=HOME_COLOR, width=2, shape="hv"), fill="tozeroy", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=tl["t"], y=lead.clip(upper=0), name=f"{game.team_b_name} leads",
        mode="lines", line=dict(color=AWAY_COLOR, width=2, shape="hv"), fill="tozeroy", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=tl["t"], y=lead, mode="lines", line=dict(width=0, shape="hv"), showlegend=False,
        customdata=pd.concat([custom, leader], axis=1).values,
        hovertemplate="%{customdata[0]}: %{customdata[1]}<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color="rgba(128,128,128,0.6)")
    _layout(fig, end, height=260)
    fig.update_yaxes(tickformat="+d")
    return fig


def quarter_points_chart(qs: pd.DataFrame, game: pd.Series) -> go.Figure:
    periods = [c for c in qs.columns if c != "Final"]
    fig = go.Figure()
    for team, color in ((game.team_a_name, HOME_COLOR), (game.team_b_name, AWAY_COLOR)):
        fig.add_trace(go.Bar(
            x=periods, y=qs.loc[team, periods], name=team, marker_color=color,
            marker_line_width=0, text=qs.loc[team, periods], textposition="outside",
            hovertemplate="%{x}: %{y} pts<extra>" + team + "</extra>",
        ))
    fig.update_layout(
        barmode="group", bargap=0.35, bargroupgap=0.08, height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_traces(marker_cornerradius=4)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", rangemode="tozero")
    return fig


# ---------------------------------------------------------------- UI
def fmt_mmss(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def game_label(g: pd.Series) -> str:
    ot = " (OT)" if g.actual_quarter > 4 else ""
    return f"#{g.game_code} · {g.team_a_name} {g.final_score_a}–{g.final_score_b} {g.team_b_name}{ot}"


def main():
    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}. Run `python src/pipeline.py` first.")
        return

    games = load_games()

    # --- sidebar: game picker
    st.sidebar.title("🏀 EuroLeague")
    seasons = sorted(games["season"].unique(), reverse=True)
    season = st.sidebar.selectbox("Season", seasons, format_func=lambda s: SEASON_LABELS.get(s, s))
    sg = games[games["season"] == season]

    teams = sorted(set(sg["team_a_name"]) | set(sg["team_b_name"]))
    team = st.sidebar.selectbox("Team", ["All teams"] + teams)
    if team != "All teams":
        sg = sg[(sg["team_a_name"] == team) | (sg["team_b_name"] == team)]

    if sg.empty:
        st.info("No games for this selection.")
        return

    sg = sg.sort_values("game_code", ascending=False)
    game_idx = st.sidebar.selectbox("Game", sg.index, format_func=lambda i: game_label(sg.loc[i]))
    game = sg.loc[game_idx]
    st.sidebar.caption(f"{len(sg)} games · data from live.euroleague.net")

    plays = load_plays(season, game.game_code)
    if plays.empty:
        st.warning("No play-by-play data for this game.")
        return

    # --- header / scoreboard
    st.caption(f"{SEASON_LABELS.get(season, season)} · Game #{game.game_code}")
    left, mid, right = st.columns([5, 3, 5])
    with left:
        st.markdown(f"### {game.team_a_name}\n<span style='color:{HOME_COLOR}'>■</span> Home · {game.team_a_code}", unsafe_allow_html=True)
    with mid:
        st.markdown(
            f"<div style='text-align:center;font-size:2.6rem;font-weight:700;line-height:1.1'>"
            f"{game.final_score_a} – {game.final_score_b}</div>"
            f"<div style='text-align:center;opacity:.7'>{'Final (OT)' if game.actual_quarter > 4 else 'Final'}</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(f"<div style='text-align:right'><h3>{game.team_b_name}</h3>{game.team_b_code} · Away "
                    f"<span style='color:{AWAY_COLOR}'>■</span></div>", unsafe_allow_html=True)

    qs = quarter_scores(plays, game)
    st.dataframe(qs, width="stretch")

    tab_overview, tab_box, tab_pbp = st.tabs(["📊 Overview", "📋 Box score", "🎬 Play-by-play"])

    # --- overview
    with tab_overview:
        fm = flow_metrics(plays, game)
        k = st.columns(4)
        k[0].metric("Lead changes", fm["lead_changes"])
        k[1].metric("Times tied", fm["ties"])
        k[2].metric(f"Largest lead · {game.team_a_code}", fm["max_lead_a"])
        k[3].metric(f"Largest lead · {game.team_b_code}", fm["max_lead_b"])
        k = st.columns(4)
        k[0].metric(f"Best run · {game.team_a_code}", f"{fm['run_a']}–0")
        k[1].metric(f"Best run · {game.team_b_code}", f"{fm['run_b']}–0")
        k[2].metric(f"Time leading · {game.team_a_code}", fmt_mmss(fm["lead_time_a"]))
        k[3].metric(f"Time leading · {game.team_b_code}", fmt_mmss(fm["lead_time_b"]))

        st.subheader("Score progression")
        st.plotly_chart(score_chart(plays, game), width="stretch")
        st.subheader("Lead")
        st.plotly_chart(lead_chart(plays, game), width="stretch")

        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Team comparison")
            sa, sb = team_stats(plays, game.team_a_code), team_stats(plays, game.team_b_code)
            comp = pd.DataFrame({game.team_a_name: sa, game.team_b_name: sb}).astype(str)
            st.dataframe(comp, width="stretch", height=458)
        with c2:
            st.subheader("Points by period")
            st.plotly_chart(quarter_points_chart(qs, game), width="stretch")
            st.subheader("Game facts")
            facts = game_facts(plays, game)
            st.dataframe(facts, hide_index=True, width="stretch", height=38 + 35 * len(facts))

    # --- box score
    with tab_box:
        for code, name, color in ((game.team_a_code, game.team_a_name, HOME_COLOR),
                                  (game.team_b_code, game.team_b_name, AWAY_COLOR)):
            st.markdown(f"#### <span style='color:{color}'>■</span> {name}", unsafe_allow_html=True)
            box = player_box(plays, code)
            if box.empty:
                st.caption("No player data.")
                continue
            totals = box.drop(columns=["#", "Player"]).sum()
            box_tot = pd.concat([box, pd.DataFrame([{"#": "", "Player": "TOTAL", **totals.to_dict()}])], ignore_index=True)
            st.dataframe(box_tot, hide_index=True, width="stretch",
                         height=min(38 + 35 * len(box_tot), 640))

    # --- play-by-play
    with tab_pbp:
        f1, f2, f3, f4 = st.columns([2, 3, 3, 2])
        periods = list(plays["period"].unique())
        sel_q = f1.multiselect("Period", periods, default=periods)
        team_opts = {"Both": None, game.team_a_name: game.team_a_code, game.team_b_name: game.team_b_code}
        sel_team = f2.selectbox("Team", list(team_opts))
        cats = list(PLAY_CATEGORIES)
        default_cats = [c for c in cats if c != "Substitutions"]
        sel_cats = f3.multiselect("Event type", cats, default=default_cats)
        scoring_only = f4.toggle("Scoring plays only")

        players = sorted(plays["player_name"].dropna().unique())
        sel_players = st.multiselect("Players", players, placeholder="All players")

        types = set().union(*(PLAY_CATEGORIES[c] for c in sel_cats)) if sel_cats else set()
        p = plays[plays["period"].isin(sel_q) & plays["play_type"].isin(types)]
        if team_opts[sel_team]:
            p = p[p["team_code"] == team_opts[sel_team]]
        if scoring_only:
            p = p[p["is_scoring_play"] == 1]
        if sel_players:
            p = p[p["player_name"].isin(sel_players)]

        view = pd.DataFrame({
            "Period": p["period"],
            "Clock": p["marker_time"],
            "Team": p["team_code"],
            "Player": p["player_name"],
            "Event": p["play_info"],
            "Type": p["play_type"],
            "Score": p["current_score_a"].astype(int).astype(str) + "–" + p["current_score_b"].astype(int).astype(str),
            "Lead": p["score_lead_a"].astype(int),
            "Pts": (p["points_scored_a"] + p["points_scored_b"]).astype(int),
        })
        st.caption(f"{len(view)} of {len(plays)} events · Lead is from {game.team_a_code}'s perspective")
        st.dataframe(
            view, hide_index=True, width="stretch", height=620,
            column_config={
                "Lead": st.column_config.NumberColumn(format="%+d"),
                "Pts": st.column_config.NumberColumn(width="small"),
            },
        )
        st.download_button(
            "Download play-by-play (CSV)",
            plays.to_csv(index=False).encode("utf-8"),
            file_name=f"{season}_game_{game.game_code}_plays.csv",
            mime="text/csv",
        )


main()
