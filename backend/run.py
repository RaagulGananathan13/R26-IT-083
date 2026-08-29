"""
Development entrypoint.

    cd backend
    python run.py                 # 127.0.0.1:8000
    python run.py --reload        # auto-reload on source change
    python run.py --warm          # load every component before serving
    python run.py --host 0.0.0.0 --port 8080

`--reload` and `--warm` together is usually a mistake: every reload repays the
full model-loading cost.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `cvxai` importable no matter where the interpreter was launched from.
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the R26-IT-083 unified backend")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true",
                        help="auto-reload on source change (development only)")
    parser.add_argument("--warm", action="store_true",
                        help="load every serviceable component at startup")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    if args.warm:
        os.environ["CVXAI_EAGER_LOAD"] = "1"
    if args.log_level:
        os.environ["CVXAI_LOG_LEVEL"] = args.log_level

    # Imported after the environment is set, so settings pick the overrides up.
    import uvicorn

    from cvxai.settings import get_settings

    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port

    if args.reload and args.warm:
        print("[run] --reload with --warm repays the full model-load cost on every "
              "source change.", file=sys.stderr)

    uvicorn.run(
        "cvxai.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
        # The request-context middleware already logs every request with its id
        # and timing; uvicorn's access logger would print a second, unformatted
        # line for the same request.
        access_log=False,
        # The components are internally serialised, so extra workers would
        # multiply memory without multiplying throughput.
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
