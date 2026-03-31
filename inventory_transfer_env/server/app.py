from inventory_transfer_env.models import InventoryTransferAction, InventoryTransferObservation
from inventory_transfer_env.server.inventory_transfer_environment import (
    InventoryTransferEnvironment,
)
from openenv.core.env_server import create_app

app = create_app(
    InventoryTransferEnvironment,
    InventoryTransferAction,
    InventoryTransferObservation,
    env_name="inventory_transfer_env",
)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "env": "inventory_transfer_env"}


def main():
    import uvicorn
    import os

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))

    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
