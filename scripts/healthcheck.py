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
