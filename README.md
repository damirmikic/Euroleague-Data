# EuroLeague Play-by-Play Dataset (2024, 2025 & 2026)

Comprehensive play-by-play and game metadata dataset for EuroLeague **2024-2025 (`E2024`)** and **2025-2026 (`E2025`)** seasons, plus the in-progress **2026-2027 (`E2026`)** season, scraped directly from the official EuroLeague Live API (`live.euroleague.net`).

---

## 📁 Dataset Directory Structure

```
Euroleague Data/
├── data/
│   ├── raw/                              # Cached raw JSON responses from EuroLeague API
│   │   ├── E2024/game_{id}.json
│   │   ├── E2025/game_{id}.json
│   │   └── E2026/game_{id}.json          # Current season (finished games only)
│   └── processed/                        # Structured tabular datasets
│       ├── euroleague_games.csv          # Combined games metadata (E2024 + E2025)
│       ├── euroleague_plays.csv          # Combined play-by-play events (E2024 + E2025)
│       ├── euroleague_games.parquet      # High-performance Parquet format
│       ├── euroleague_plays.parquet      # High-performance Parquet format
│       ├── euroleague.db                 # Indexed SQLite database
│       ├── E2024_games.csv               # Season-specific splits
│       ├── E2024_plays.csv
│       ├── E2025_games.csv
│       └── E2025_plays.csv
├── src/
│   ├── scraper.py                        # Resilient scraper with rate-limiting & backoff
│   ├── parser.py                         # Data transformer & enriched metrics
│   ├── pipeline.py                       # Master CLI runner
│   └── fetch_current_season.py           # Incremental fetcher for the current season (E2026)
├── app.py                                # Streamlit game dashboard
├── requirements.txt
└── README.md
```

---

## 📊 Data Dictionary & Schemas

### 1. `games` Table (`euroleague_games.csv` / `euroleague.db:games`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `season` | string | Season code (e.g. `E2024`, `E2025`) |
| `game_code` | integer | Unique game identifier within season (e.g. `1` to `335`) |
| `team_a_code` | string | Home team 3-letter code (e.g. `BER`, `PAN`, `MAD`, `TEL`) |
| `team_a_name` | string | Home team official name |
| `team_b_code` | string | Away team 3-letter code |
| `team_b_name` | string | Away team official name |
| `actual_quarter` | integer | Total periods played (`4` = regulation, `5+` = Overtime) |
| `final_score_a` | integer | Final points scored by Team A |
| `final_score_b` | integer | Final points scored by Team B |
| `total_plays` | integer | Total event records logged in the game |
| `is_live` | boolean | Game live status flag |

### 2. `plays` Table (`euroleague_plays.csv` / `euroleague.db:plays`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `season` | string | Season code (`E2024`, `E2025`) |
| `game_code` | integer | Game identifier |
| `quarter_num` | integer | Period number (`1`, `2`, `3`, `4`, `5` for OT1, `6` for OT2, etc.) |
| `quarter_name` | string | Period label (`Q1`, `Q2`, `Q3`, `Q4`, `OT`) |
| `play_number` | integer | Chronological play index sequence |
| `play_type` | string | Event code (e.g. `2FGM`, `2FGA`, `3FGM`, `3FGA`, `FTM`, `FTA`, `D`, `O`, `TO`, `AS`, `ST`, `BLK`, `FOUL`, `IN`, `OUT`, `BP`, `EP`) |
| `play_info` | string | Event description (e.g. `Two Pointer (1/1 - 2 pt)`, `Def Rebound (4)`) |
| `team_code` | string | Action team code |
| `team_name` | string | Action team name |
| `player_id` | string | Player ID code (e.g. `P002328`) |
| `player_name` | string | Player name (e.g. `LESSORT, MATHIAS`) |
| `dorsal` | string | Jersey number |
| `minute` | integer | Elapsed game minute (`1` to `40+`) |
| `marker_time` | string | Quarter clock timestamp (`MM:SS`) |
| `seconds_remaining_in_quarter` | integer | Seconds left in current quarter |
| `elapsed_seconds_in_game` | integer | Total game seconds elapsed |
| `current_score_a` | integer | Running cumulative score of Team A at this event |
| `current_score_b` | integer | Running cumulative score of Team B at this event |
| `points_scored_a` | integer | Points scored by Team A on this exact play (`0`, `1`, `2`, `3`) |
| `points_scored_b` | integer | Points scored by Team B on this exact play (`0`, `1`, `2`, `3`) |
| `score_lead_a` | integer | Score differential (`current_score_a - current_score_b`) |
| `is_scoring_play` | boolean | `True` if play resulted in points scored |
| `raw_type` | integer | Raw event category code from EuroLeague API |
| `comment` | string | Additional metadata / review comments |

---

## 🚀 Quickstart & Usage

### 1. Python / Pandas

```python
import pandas as pd

# Load master datasets
games_df = pd.read_parquet("data/processed/euroleague_games.parquet")
plays_df = pd.read_parquet("data/processed/euroleague_plays.parquet")

# Example: Top 10 scorers in 2024 season
points_per_player = (
    plays_df[plays_df["season"] == "E2024"]
    .groupby(["player_id", "player_name", "team_name"])["points_scored_a", "points_scored_b"]
    .apply(lambda df: (df["points_scored_a"] + df["points_scored_b"]).sum())
    .reset_index(name="total_points")
    .sort_values(by="total_points", ascending=False)
)
print(points_per_player.head(10))
```

### 2. SQLite Database Queries

```sql
-- Connect to data/processed/euroleague.db

-- Average points scored per game by season
SELECT 
    season,
    COUNT(*) AS total_games,
    ROUND(AVG(final_score_a + final_score_b), 1) AS avg_total_points,
    ROUND(AVG(final_score_a), 1) AS avg_home_score,
    ROUND(AVG(final_score_b), 1) AS avg_away_score
FROM games
GROUP BY season;

-- Clutch plays (last 2 minutes of Q4 or OT with score diff <= 5)
SELECT 
    season,
    game_code,
    quarter_name,
    marker_time,
    player_name,
    play_type,
    play_info,
    score_lead_a
FROM plays
WHERE (quarter_num >= 4)
  AND (seconds_remaining_in_quarter <= 120)
  AND (ABS(score_lead_a) <= 5)
  AND (is_scoring_play = 1)
ORDER BY season, game_code, elapsed_seconds_in_game;
```

---

## 🖥️ Game Dashboard

Interactive dashboard to pick a game and explore it:

```bash
streamlit run app.py
```

- **Game picker** (sidebar): season → team filter → game
- **Scoreboard** with period-by-period scores (overtimes split into OT1, OT2, …)
- **Overview**: lead changes, ties, largest leads, best runs, time leading, score progression & lead charts, team comparison, points by period, betting facts (first/last points, race to 10/20/30, half-time, margin)
- **Box score**: per-player points, shooting splits, rebounds, assists, steals, turnovers, blocks, fouls
- **Play-by-play**: filter by period, team, event type, player or scoring plays; CSV download
- **🔄 Fetch new games** (sidebar): downloads newly finished current-season games, rebuilds the datasets and reloads the dashboard in one click. The dashboard also reloads automatically whenever `euroleague.db` is rebuilt from the command line.

## 🔄 Updating / Re-running the Pipeline

### Current season (2026-27)

```bash
# Fetch newly finished E2026 games and rebuild all processed datasets
python src/fetch_current_season.py
```

The current season (`CURRENT_SEASON` in `src/scraper.py`) is fetched in *ongoing* mode:
unplayed and live games are never cached, so re-running after each round picks up new
results. Scraping stops at the first run of 10 empty game codes.

### Full pipeline

```bash
# Update all seasons (E2024, E2025, E2026)
python src/pipeline.py

# Force re-download all games
python src/pipeline.py --force-scrape

# Fast re-process cached JSONs without re-downloading
python src/pipeline.py --skip-scrape
```
