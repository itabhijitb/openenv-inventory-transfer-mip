import os

from openenv.core.env_server import create_app

from inventory_transfer_env.models import InventoryTransferAction, InventoryTransferObservation
from inventory_transfer_env.server.inventory_transfer_environment import (
    InventoryTransferEnvironment,
)

app = create_app(
    InventoryTransferEnvironment,
    InventoryTransferAction,
    InventoryTransferObservation,
    env_name="inventory_transfer_env",
)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "env": "inventory_transfer_env"}


@app.get("/debug/env")
def debug_env() -> dict:
    return {
        "has_API_BASE_URL": bool(os.environ.get("API_BASE_URL")),
        "has_MODEL_NAME": bool(os.environ.get("MODEL_NAME")),
        "has_HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
    }


def main():
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))

    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
