# scripts/healthcheck.py (A1).
# It calls http://127.0.0.1:$PORT/health with an 8-second timeout.
# It exits 0 on a 200 and 1 otherwise.
# Docker runs it every 30 seconds and shows (healthy) in docker ps.
# It uses Python's urllib, so the slim image needs no curl.
# It reads PORT, so it works with -e PORT=10000.

# healthcheck.py uses plain Python, with a 180 s start period.
# The slim image has no curl. The start period tells Docker to ignore failures
# during the first model load.

"""Docker healthcheck: exit 0 when GET /health answers 200, otherwise exit 1."""

from __future__ import annotations

import logging
import os
import sys
import urllib.request

logger = logging.getLogger("reviewpulse.healthcheck")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    url = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/health"
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            if response.status == 200:
                return 0
            logger.warning("%s answered %s", url, response.status)
    except Exception as error:
        logger.warning("%s failed: %s", url, error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
