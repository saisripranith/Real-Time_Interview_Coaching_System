"""
Entry point to run the FastAPI server in a cross-platform way.

This file can be executed directly on Windows, macOS or Linux:

  python main.py

If you prefer to use the `uv` CLI (if installed), the included `start.sh`
will try to use it and otherwise fall back to running this file.
"""
import os
import multiprocessing

import uvicorn


def main() -> None:
    """Run the FastAPI app with sensible defaults.

    This uses uvicorn programmatic API which works cross-platform.
    """
    # Allow overriding host/port via env vars for flexibility
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", "8000"))

    # Enable reload if explicitly requested (development only)
    reload_flag = os.environ.get("RELOAD", "true").lower() in ("1", "true", "yes")

    # Use number of workers only when not reloading (uvicorn reload not compatible with multiple workers)
    workers = 1
    if not reload_flag:
        try:
            workers = int(os.environ.get("UVICORN_WORKERS", str(max(1, multiprocessing.cpu_count() // 2))))
        except Exception:
            workers = 1

    # When reload is enabled, must use import string; otherwise can use app object
    if reload_flag:
        uvicorn.run(
            "server:app",  # Import string for reload support
            host=host,
            port=port,
            reload=True,
            log_level=os.environ.get("LOG_LEVEL", "info"),
        )
    else:
        from server import app
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers,
            log_level=os.environ.get("LOG_LEVEL", "info"),
        )


if __name__ == "__main__":
    main()
