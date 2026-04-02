import logging
import os

from openenv.core.env_server import create_app

from inventory_transfer_env.models import InventoryTransferAction, InventoryTransferObservation
from inventory_transfer_env.server.inventory_transfer_environment import (
    InventoryTransferEnvironment,
)

logger = logging.getLogger(__name__)

app = create_app(
    InventoryTransferEnvironment,
    InventoryTransferAction,
    InventoryTransferObservation,
    env_name="inventory_transfer_env",
)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "env": "inventory_transfer_env"}


_debug_enabled = os.environ.get("ENABLE_DEBUG_ENDPOINTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if _debug_enabled:

    @app.get("/debug/env")
    def debug_env() -> dict:
        return {
            "has_API_BASE_URL": bool(os.environ.get("API_BASE_URL")),
            "has_MODEL_NAME": bool(os.environ.get("MODEL_NAME")),
            "has_HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
        }


def _get_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"Invalid integer for env var {name}={raw!r}") from e


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = _get_int_env("PORT", 8000)
    workers = _get_int_env("WORKERS", 1)

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    logger.info(
        "Starting server host=%s port=%s workers=%s debug_endpoints=%s",
        host,
        port,
        workers,
        _debug_enabled,
    )

    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
