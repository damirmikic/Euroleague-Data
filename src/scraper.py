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

    def fetch_game(self, season: str, gamecode: int, force: bool = False) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Fetches play-by-play data for a game.
        Returns (data_dict, is_valid_game).
        """
        cache_file = self._get_cache_path(season, gamecode)
        if not force and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and data.get("TeamA") is not None:
                        return data, True
                    elif isinstance(data, dict) and data.get("empty", False):
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
                        # Empty response indicates non-existent game
                        self._save_empty(cache_file)
                        return None, False

                    try:
                        data = resp.json()
                    except json.JSONDecodeError:
                        self._save_empty(cache_file)
                        return None, False

                    if isinstance(data, dict) and data.get("TeamA"):
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        return data, True
                    else:
                        self._save_empty(cache_file)
                        return None, False

                elif resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    logger.warning(f"Rate limited (429). Sleeping for {retry_after + 2}s before retry...")
                    time.sleep(retry_after + 2)
                    continue

                elif resp.status_code in (404, 400, 204):
                    self._save_empty(cache_file)
                    return None, False

                else:
                    logger.warning(f"Unexpected status {resp.status_code} for {season} game {gamecode} (attempt {attempt+1}/{max_retries})")
                    time.sleep(2 ** attempt)

            except (requests.RequestException, Exception) as e:
                logger.warning(f"Network error on {season} game {gamecode}: {e} (attempt {attempt+1}/{max_retries})")
                time.sleep(2 ** attempt)

        return None, False

    def _save_empty(self, cache_file: Path):
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"empty": True}, f)

    def fetch_season(self, season: str, max_games: int = 360, consecutive_empty_limit: int = 10, force: bool = False):
        """
        Fetches all games for a season sequentially until consecutive_empty_limit non-existent games are encountered.
        """
        logger.info(f"--- Starting scraping for season {season} ---")
        consecutive_empty = 0
        valid_count = 0

        for gamecode in range(1, max_games + 1):
            data, is_valid = self.fetch_game(season, gamecode, force=force)
            if is_valid:
                consecutive_empty = 0
                valid_count += 1
                if valid_count % 25 == 0 or gamecode <= 5:
                    logger.info(f"[{season}] Game {gamecode:3d}: {data.get('TeamA')} vs {data.get('TeamB')}")
            else:
                consecutive_empty += 1
                if consecutive_empty >= consecutive_empty_limit and gamecode > 300:
                    logger.info(f"[{season}] Reached {consecutive_empty} consecutive empty games at game {gamecode}. Finished season.")
                    break

        logger.info(f"--- Finished {season}: {valid_count} games collected ---")
        return valid_count
