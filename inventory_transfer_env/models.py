from __future__ import annotations

from openenv.core.env_server.interfaces import Action, Observation, State
from pydantic import BaseModel, Field


class Warehouse(BaseModel):
    id: str
    inventory: dict[str, int]
    demand: dict[str, int]


class Transfer(BaseModel):
    from_warehouse: str
    to_warehouse: str
    product: str
    quantity: int = Field(ge=0)


class InventoryTransferAction(Action):
    transfers: list[Transfer] = Field(default_factory=list)


class InventoryTransferObservation(Observation):
    task_id: str
    warehouses: list[Warehouse]
    products: list[str]
    transfer_cost: dict[str, dict[str, float]]
    lane_fixed_cost: dict[str, dict[str, float]] | None = None
    penalty_per_unit_shortage: float
    budget: float | None = None
    outbound_capacity: dict[str, int] | None = None
    inbound_capacity: dict[str, int] | None = None

    sku_capacity: dict[str, dict[str, int]] | None = None
    lane_capacity: dict[str, dict[str, dict[str, int]]] | None = None
    min_transfer_lot: dict[str, int] | None = None

    demand_noise_pct: float | None = None
    max_steps: int = 1
    step_number: int = 0
    per_step_budget: float | None = None

    transfer_cost_total: float = 0.0
    lane_activation_cost_total: float = 0.0
    shortage_units_total: int = 0
    shortage_penalty_total: float = 0.0
    total_cost: float = 0.0

    lanes_activated: int = 0
    co2_kg: float = 0.0

    total_demand_units: int = 0
    fulfilled_units: int = 0
    fill_rate: float = 0.0

    optimal_cost: float | None = None
    optimal_feasible: bool | None = None

    score: float | None = None
    disqualified: bool = False
    dq_reasons: list[str] = Field(default_factory=list)
    optimality_ratio: float | None = None
    cost_gap: float | None = None
    is_feasible: bool = True
    violations: list[str] = Field(default_factory=list)


class InventoryTransferState(State):
    task_id: str = ""
    current_inventory: dict[str, dict[str, int]] = Field(default_factory=dict)
    cumulative_transfer_cost: float = 0.0
    cumulative_lane_activation_cost: float = 0.0
    steps_remaining: int = 1
