"""Start HydroGelCam2.

    python run_server.py                 # http://127.0.0.1:8000
    python run_server.py --port 8123     # a different port
    PORT=8123 python run_server.py       # same, from the environment

The PORT environment variable is honoured so several instances can run side by
side without colliding -- handy when one is already open on the default port.
"""

from __future__ import annotations

import argparse
import os

import uvicorn

DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HydroGelCam2 server.")
    parser.add_argument("--host", default=os.environ.get("HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", DEFAULT_PORT)))
    parser.add_argument("--reload", action="store_true",
                        help="restart on source changes, for development")
    args = parser.parse_args()

    print(f"HydroGelCam2 -> http://{args.host}:{args.port}")
    uvicorn.run("app.main:app", host=args.host, port=args.port,
                reload=args.reload)


if __name__ == "__main__":
    main()
