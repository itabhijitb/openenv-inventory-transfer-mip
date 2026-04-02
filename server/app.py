from inventory_transfer_env.server.app import app

__all__ = ["app"]


def main() -> None:
    import os

    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))

    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
