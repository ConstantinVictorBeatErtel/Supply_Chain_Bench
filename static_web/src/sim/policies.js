export function initialForecast(spec, role) {
  if (spec.tier === 1) return 8;
  if (spec.topology === "y" && !["retailer_a", "retailer_b"].includes(role)) return 15;
  return 7.5;
}

export class AdaptiveBaseStockPolicy {
  constructor(spec, role) {
    this.forecast = initialForecast(spec, role);
    this.alpha = 0.25;
    this.orderCap = spec.order_cap;
    this.replenishmentDelay = spec.order_delay + spec.shipment_delay;
    this.policy_id = "adaptive_base_stock";
    this.policy_version = "2";
  }

  act(observation) {
    const demand = Number(observation.state.incoming_demand_or_order);
    this.forecast = (1 - this.alpha) * this.forecast + this.alpha * demand;
    const target = Math.ceil(this.replenishmentDelay * this.forecast);
    const position = observation.state.inventory_position;
    return Math.min(this.orderCap, Math.max(0, target - position));
  }
}

export class ScarcityAggressivePolicy {
  constructor(base) {
    this.base = base;
    this.increment = 8;
    this.policy_id = "scarcity_aggressive";
    this.policy_version = "1";
  }

  act(observation) {
    return Math.min(this.base.orderCap, this.base.act(observation) + this.increment);
  }
}

export class DemandTrackingScarcityPolicy {
  constructor({ increment = 8, orderCap = 128 } = {}) {
    this.increment = increment;
    this.orderCap = orderCap;
    this.policy_id = "demand_tracking_scarcity";
    this.policy_version = "1";
  }

  act(observation) {
    const demand = Number(observation.state.incoming_demand_or_order);
    return Math.min(this.orderCap, Math.max(0, Math.trunc(demand) + this.increment));
  }
}

export function adaptivePolicy(spec, role) {
  return new AdaptiveBaseStockPolicy(spec, role);
}

export function counterpartyPolicies(spec, controlledRole) {
  return Object.fromEntries(spec.roles.filter((role) => role !== controlledRole).map((role) => {
    const base = adaptivePolicy(spec, role);
    if (spec.counterparty_policy === "demand_tracking_scarcity_v1"
        && ["retailer_a", "retailer_b"].includes(role)) {
      return [role, new DemandTrackingScarcityPolicy({ orderCap: spec.order_cap })];
    }
    return [
      role,
      spec.aggressive_retailers && ["retailer_a", "retailer_b"].includes(role)
        ? new ScarcityAggressivePolicy(base)
        : base,
    ];
  }));
}
