import os
import json
import time
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("euroleague_scraper")

BASE_URL = "https://live.euroleague.net/api/PlaybyPlay"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.euroleaguebasketball.net/",
    "Origin": "https://www.euroleaguebasketball.net",
}

# Season currently in progress (2026-27). Games for this season are fetched in
# "ongoing" mode: unplayed/live games are never cached, so they get re-checked.
CURRENT_SEASON = "E2026"

def is_game_complete(data: Dict[str, Any]) -> bool:
    """A game is complete when it is not live and has 4th quarter play-by-play."""
    if str(data.get("Live", "False")).lower() == "true":
        return False
    q4 = data.get("ForthQuarter")
    return isinstance(q4, list) and len(q4) > 0

class EuroleagueScraper:
    def __init__(self, raw_dir: str = "data/raw", request_delay: float = 0.4):
        self.raw_dir = Path(raw_dir)
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _get_cache_path(self, season: str, gamecode: int) -> Path:
        season_dir = self.raw_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)
        return season_dir / f"game_{gamecode}.json"

    def fetch_game(self, season: str, gamecode: int, force: bool = False, ongoing: bool = False) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Fetches play-by-play data for a game.
        Returns (data_dict, is_valid_game).

        With ongoing=True (season in progress), empty responses are not cached and
        live/unfinished games are neither cached nor returned as valid.
        """
        cache_file = self._get_cache_path(season, gamecode)
        if not force and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and data.get("TeamA") is not None:
                        if not ongoing or is_game_complete(data):
                            return data, True
                    elif isinstance(data, dict) and data.get("empty", False) and not ongoing:
                        return None, False
            except Exception:
                pass  # Re-download if corrupted

        params = {"gamecode": gamecode, "seasoncode": season}
        max_retries = 5

        for attempt in range(max_retries):
            try:
                time.sleep(self.request_delay)
                resp = self.session.get(BASE_URL, params=params, timeout=15)

                if resp.status_code == 200:
                    text = resp.text.strip()
                    if not text:
                        # Empty response indicates non-existent (or not yet played) game
                        self._save_empty(cache_file, ongoing)
                        return None, False

                    try:
                        data = resp.json()
                    except json.JSONDecodeError:
                        self._save_empty(cache_file, ongoing)
                        return None, False

                    if isinstance(data, dict) and data.get("TeamA"):
                        if ongoing and not is_game_complete(data):
                            # Live or not started yet - don't cache partial data
                            self._drop_cache(cache_file)
                            return None, False
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        return data, True
                    else:
                        self._save_empty(cache_file, ongoing)
                        return None, False

                elif resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    logger.warning(f"Rate limited (429). Sleeping for {retry_after + 2}s before retry...")
                    time.sleep(retry_after + 2)
                    continue

                elif resp.status_code in (404, 400, 204):
                    self._save_empty(cache_file, ongoing)
                    return None, False

                else:
                    logger.warning(f"Unexpected status {resp.status_code} for {season} game {gamecode} (attempt {attempt+1}/{max_retries})")
                    time.sleep(2 ** attempt)

            except (requests.RequestException, Exception) as e:
                logger.warning(f"Network error on {season} game {gamecode}: {e} (attempt {attempt+1}/{max_retries})")
                time.sleep(2 ** attempt)

        return None, False

    def _save_empty(self, cache_file: Path, ongoing: bool = False):
        if ongoing:
            # Game may be played later - make sure it gets re-checked next run
            self._drop_cache(cache_file)
            return
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"empty": True}, f)

    def _drop_cache(self, cache_file: Path):
        if cache_file.exists():
            cache_file.unlink()

    def fetch_season(self, season: str, max_games: int = 360, consecutive_empty_limit: int = 10, force: bool = False, ongoing: Optional[bool] = None):
        """
        Fetches all games for a season sequentially until consecutive_empty_limit non-existent games are encountered.
        ongoing defaults to True for CURRENT_SEASON: stops at the first run of empty games
        (end of played schedule) instead of probing up to game 300.
        """
        if ongoing is None:
            ongoing = season == CURRENT_SEASON
        logger.info(f"--- Starting scraping for season {season}{' (ongoing)' if ongoing else ''} ---")
        consecutive_empty = 0
        valid_count = 0

        for gamecode in range(1, max_games + 1):
            data, is_valid = self.fetch_game(season, gamecode, force=force, ongoing=ongoing)
            if is_valid:
                consecutive_empty = 0
                valid_count += 1
                if valid_count % 25 == 0 or gamecode <= 5:
                    logger.info(f"[{season}] Game {gamecode:3d}: {data.get('TeamA')} vs {data.get('TeamB')}")
            else:
                consecutive_empty += 1
                if consecutive_empty >= consecutive_empty_limit and (ongoing or gamecode > 300):
                    logger.info(f"[{season}] Reached {consecutive_empty} consecutive empty games at game {gamecode}. Finished season.")
                    break

        logger.info(f"--- Finished {season}: {valid_count} games collected ---")
        return valid_count
