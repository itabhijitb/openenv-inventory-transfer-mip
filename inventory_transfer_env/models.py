from __future__ import annotations

from typing import Dict, List, Optional

from openenv.core.env_server.interfaces import Action, Observation, State
from pydantic import BaseModel, Field


class Warehouse(BaseModel):
    id: str
    inventory: Dict[str, int]
    demand: Dict[str, int]


class Transfer(BaseModel):
    from_warehouse: str
    to_warehouse: str
    product: str
    quantity: int = Field(ge=0)


class InventoryTransferAction(Action):
    transfers: List[Transfer] = Field(default_factory=list)


class InventoryTransferObservation(Observation):
    task_id: str
    warehouses: List[Warehouse]
    products: List[str]
    transfer_cost: Dict[str, Dict[str, float]]
    lane_fixed_cost: Optional[Dict[str, Dict[str, float]]] = None
    penalty_per_unit_shortage: float
    budget: Optional[float] = None
    outbound_capacity: Optional[Dict[str, int]] = None
    inbound_capacity: Optional[Dict[str, int]] = None

    sku_capacity: Optional[Dict[str, Dict[str, int]]] = None
    lane_capacity: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None
    min_transfer_lot: Optional[Dict[str, int]] = None

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

    optimal_cost: Optional[float] = None
    optimal_feasible: Optional[bool] = None

    score: Optional[float] = None
    disqualified: bool = False
    dq_reasons: List[str] = Field(default_factory=list)
    optimality_ratio: Optional[float] = None
    cost_gap: Optional[float] = None
    is_feasible: bool = True
    violations: List[str] = Field(default_factory=list)


class InventoryTransferState(State):
    task_id: str = ""
