"""
Incremental fetcher for the season in progress (2026-27, E2026).

Fetches newly finished games of the current season, then rebuilds the combined
processed datasets (CSV / Parquet / SQLite) from cache for all seasons.
Safe to run repeatedly (e.g. after each round): finished games are cached,
unplayed or live games are re-checked on the next run.

    python src/fetch_current_season.py
"""
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scraper import EuroleagueScraper, CURRENT_SEASON
from src.pipeline import run_pipeline, DEFAULT_SEASONS

logger = logging.getLogger("euroleague_current_season")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Fetch new games for the current season ({CURRENT_SEASON})")
    parser.add_argument("--season", default=CURRENT_SEASON, help=f"Season code to update (default: {CURRENT_SEASON})")
    parser.add_argument("--delay", type=float, default=0.35, help="Delay in seconds between requests (default: 0.35)")
    parser.add_argument("--max-games", type=int, default=350, help="Max gamecode to probe (default: 350)")
    parser.add_argument("--no-rebuild", action="store_true", help="Only fetch raw JSON, skip rebuilding processed datasets")
    args = parser.parse_args()

    scraper = EuroleagueScraper(raw_dir="data/raw", request_delay=args.delay)
    scraper.fetch_season(args.season, max_games=args.max_games, ongoing=True)

    if not args.no_rebuild:
        seasons = DEFAULT_SEASONS if args.season in DEFAULT_SEASONS else DEFAULT_SEASONS + [args.season]
        run_pipeline(seasons=seasons, skip_scrape=True)
