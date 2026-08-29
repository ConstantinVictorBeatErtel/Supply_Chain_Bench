import { deepCopy, pythonRound, stableStringify } from "./canonical.js";
import { PythonRandom } from "./python_random.js";
import {
  negativeBinomialR10P05, researchDigest, researchPoisson, researchUniform,
} from "./research_random.js";
import { deriveSeed } from "./scenario.js";

export function topology(spec) {
  if (spec.topology === "serial") {
    return {
      upstream: {
        retailer: "wholesaler", wholesaler: "distributor",
        distributor: "factory", factory: null,
      },
      downstream: {
        retailer: [], wholesaler: ["retailer"],
        distributor: ["wholesaler"], factory: ["distributor"],
      },
      customers: ["retailer"],
    };
  }
  return {
    upstream: {
      retailer_a: "wholesaler", retailer_b: "wholesaler",
      wholesaler: "distributor", distributor: "factory", factory: null,
    },
    downstream: {
      retailer_a: [], retailer_b: [], wholesaler: ["retailer_a", "retailer_b"],
      distributor: ["wholesaler"], factory: ["distributor"],
    },
    customers: ["retailer_a", "retailer_b"],
  };
}

export class DemandGenerator {
  constructor(spec) {
    this.spec = spec;
    this.rng = new PythonRandom(deriveSeed(spec.master_seed_hex, "demand"));
    const params = spec.demand_parameters;
    this.x = Number(params.x0 || params.mu || params.mu_before || 0);
    this.common = Number(params.common0 || 0);
    this.researchLambda = null;
    this.burstStart = null;
    if (spec.demand_process === "episode_randomized_y_poisson_v1") {
      const demandSeed = String(params.demand_seed || spec.master_seed_hex);
      this.researchLambda = Number(params.lambda_low ?? 2)
        + (Number(params.lambda_high ?? 8) - Number(params.lambda_low ?? 2))
          * researchUniform(demandSeed, "episode/lambda");
    }
    if (spec.demand_process === "burst_collapse_y_poisson_v1") {
      const digest = researchDigest(spec.master_seed_hex, "burst-window");
      this.burstStart = 8 + ((digest[0] << 8) | digest[1]) % 9;
    }
  }

  sample(week, customers) {
    const process = this.spec.demand_process;
    const params = this.spec.demand_parameters;
    if (process === "constant_v1") return { [customers[0]]: Number(params.value) };
    if (process === "ar1_v1" || process === "shifted_ar1_v1") {
      let mu = Number(params.mu || params.mu_before || 0);
      if (params.shift_week !== undefined && params.shift_week !== null && week >= params.shift_week) {
        mu = Number(params.mu_after);
      }
      this.x = mu + Number(params.phi) * (this.x - mu) + this.rng.gauss(0, Number(params.sigma));
      return { [customers[0]]: Math.max(0, pythonRound(this.x)) };
    }
    if (process === "correlated_y_ar1_v1") {
      this.common = Number(params.phi) * this.common + this.rng.gauss(0, Number(params.sigma_common));
      return Object.fromEntries(customers.map((role) => [
        role,
        Math.max(0, pythonRound(
          Number(params.mu)
          + this.common
          + this.rng.gauss(0, Number(params.sigma_idiosyncratic)),
        )),
      ]));
    }
    if (process === "episode_randomized_y_poisson_v1") {
      const demandSeed = String(params.demand_seed || this.spec.master_seed_hex);
      return Object.fromEntries(customers.map((role) => [
        role,
        researchPoisson(
          demandSeed,
          `customer-demand/week-${week}/${role}`,
          this.researchLambda,
        ),
      ]));
    }
    if (process === "canonical_y_step_4_to_8_v1") {
      return Object.fromEntries(customers.map((role) => [role, week < 5 ? 4 : 8]));
    }
    if (process === "overdispersed_y_nb_mean10_var20_v1") {
      return Object.fromEntries(customers.map((role, roleIndex) => [
        role,
        negativeBinomialR10P05(
          this.spec.master_seed_hex,
          `${process}/week-${week}/${role}`,
          roleIndex,
        ),
      ]));
    }
    if (process === "burst_collapse_y_poisson_v1") {
      return Object.fromEntries(customers.map((role, roleIndex) => {
        const burst = this.burstStart <= week && week < this.burstStart + 4;
        const collapse = this.burstStart + 4 <= week && week < this.burstStart + 8;
        const rate = burst ? 12 : collapse ? 1 : 4;
        return [
          role,
          researchPoisson(
            this.spec.master_seed_hex,
            `${process}/week-${week}/${role}`,
            rate,
            roleIndex,
          ),
        ];
      }));
    }
    throw new RangeError(`unsupported demand process '${process}'`);
  }
}

export function allocate(requested, available, rationing) {
  const entries = Object.entries(requested);
  const total = entries.reduce((sum, [, value]) => sum + value, 0);
  if (available <= 0 || total <= 0) {
    return Object.fromEntries(entries.map(([key]) => [key, 0]));
  }
  if (available >= total) return { ...requested };
  if (rationing === "uniform") {
    const output = Object.fromEntries(entries.map(([key]) => [key, 0]));
    let remaining = available;
    const keys = entries.map(([key]) => key).sort();
    while (remaining > 0) {
      let progressed = false;
      for (const key of keys) {
        if (remaining === 0) break;
        if (output[key] < requested[key]) {
          output[key] += 1;
          remaining -= 1;
          progressed = true;
        }
      }
      if (!progressed) break;
    }
    return output;
  }

  const raw = Object.fromEntries(entries.map(([key, value]) => [key, available * value / total]));
  const output = Object.fromEntries(entries.map(([key, value]) => [key, Math.min(value, Math.trunc(raw[key]))]));
  let remaining = available - Object.values(output).reduce((sum, value) => sum + value, 0);
  const order = entries.map(([key]) => key).sort((left, right) => {
    const leftFraction = raw[left] - Math.trunc(raw[left]);
    const rightFraction = raw[right] - Math.trunc(raw[right]);
    if (leftFraction !== rightFraction) return rightFraction - leftFraction;
    if (left === right) return 0;
    return left < right ? 1 : -1;
  });
  while (remaining > 0) {
    let progressed = false;
    for (const key of order) {
      if (remaining === 0) break;
      if (output[key] < requested[key]) {
        output[key] += 1;
        remaining -= 1;
        progressed = true;
      }
    }
    if (!progressed) break;
  }
  return output;
}

function stateSnapshot(state) {
  return {
    inventory: state.inventory,
    backlog: state.backlog,
    shipment_pipeline: [...state.shipment_pipeline],
    order_pipelines: Object.fromEntries(
      Object.entries(state.order_pipelines).sort(([a], [b]) => a.localeCompare(b))
        .map(([key, value]) => [key, [...value]]),
    ),
    claimant_backlog: Object.fromEntries(
      Object.entries(state.claimant_backlog).sort(([a], [b]) => a.localeCompare(b)),
    ),
    last_order_placed: state.last_order_placed,
  };
}

export class BeerGameCore {
  constructor(spec) {
    this.spec = spec;
    const shape = topology(spec);
    this.upstream = shape.upstream;
    this.downstream = shape.downstream;
    this.customers = shape.customers;
    this.demand = new DemandGenerator(spec);
    this.week = 0;
    this.prepared = null;
    this.states = {};
    for (const role of spec.roles) {
      const claimants = this.customers.includes(role) ? [] : this.downstream[role];
      this.states[role] = {
        inventory: spec.initial_inventory,
        backlog: 0,
        shipment_pipeline: Array(spec.shipment_delay).fill(spec.initial_shipment_pipeline),
        order_pipelines: Object.fromEntries(
          claimants.map((claimant) => [
            claimant, Array(spec.order_delay).fill(spec.initial_order_pipeline),
          ]),
        ),
        claimant_backlog: Object.fromEntries(claimants.map((claimant) => [claimant, 0])),
        last_order_placed: 0,
      };
    }
  }

  virtualCustomer(role) {
    return `customer:${role}`;
  }

  onOrder(role) {
    const state = this.states[role];
    const shipments = state.shipment_pipeline.reduce((sum, value) => sum + value, 0);
    const upstream = this.upstream[role];
    if (upstream === null) return shipments;
    const upstreamState = this.states[upstream];
    const pipeline = upstreamState.order_pipelines[role] || [];
    return shipments
      + pipeline.reduce((sum, value) => sum + value, 0)
      + (upstreamState.claimant_backlog[role] || 0);
  }

  inventoryPosition(role) {
    const state = this.states[role];
    return state.inventory - state.backlog + this.onOrder(role);
  }

  prepareWeek({ operational = true } = {}) {
    if (this.prepared !== null) throw new Error("current week is already prepared");
    const week = this.week + 1;
    const received = {};
    for (const role of this.spec.roles) {
      const pipeline = this.states[role].shipment_pipeline;
      received[role] = pipeline.length ? pipeline.shift() : 0;
    }
    const customerDemand = operational
      ? this.demand.sample(week, this.customers)
      : Object.fromEntries(this.customers.map((role) => [role, 0]));
    const incomingByClaimant = {};
    const backlogBeforeByClaimant = {};
    for (const role of this.spec.roles) {
      const state = this.states[role];
      if (this.customers.includes(role)) {
        const claimant = this.virtualCustomer(role);
        incomingByClaimant[role] = { [claimant]: customerDemand[role] };
        backlogBeforeByClaimant[role] = { [claimant]: state.backlog };
      } else {
        incomingByClaimant[role] = {};
        for (const [claimant, pipeline] of Object.entries(state.order_pipelines).sort(([a], [b]) => a.localeCompare(b))) {
          incomingByClaimant[role][claimant] = pipeline.length ? pipeline.shift() : 0;
        }
        backlogBeforeByClaimant[role] = { ...state.claimant_backlog };
      }
    }

    const allocations = {};
    for (const role of this.spec.roles) {
      const state = this.states[role];
      const requested = Object.fromEntries(
        Object.entries(incomingByClaimant[role]).map(([claimant, quantity]) => [
          claimant, (backlogBeforeByClaimant[role][claimant] || 0) + quantity,
        ]),
      );
      const available = state.inventory + received[role];
      const allocation = allocate(requested, available, this.spec.rationing);
      allocations[role] = allocation;
      state.inventory = available - Object.values(allocation).reduce((sum, value) => sum + value, 0);
      const remaining = Object.fromEntries(
        Object.entries(requested).map(([claimant, quantity]) => [
          claimant, quantity - (allocation[claimant] || 0),
        ]),
      );
      state.backlog = Object.values(remaining).reduce((sum, value) => sum + value, 0);
      state.claimant_backlog = this.customers.includes(role) ? {} : remaining;
      if (state.inventory < 0 || state.backlog < 0) throw new Error("negative physical state");
      if (state.inventory && state.backlog) throw new Error("inventory and backlog cannot both be positive");
    }

    for (const supplier of this.spec.roles) {
      for (const [claimant, quantity] of Object.entries(allocations[supplier])) {
        if (claimant in this.states) this.states[claimant].shipment_pipeline.push(quantity);
      }
    }

    const localCosts = Object.fromEntries(this.spec.roles.map((role) => [
      role,
      this.spec.holding_cost * this.states[role].inventory
      + this.spec.backlog_cost * this.states[role].backlog,
    ]));
    this.prepared = {
      week, operational, received, incoming_by_claimant: incomingByClaimant,
      backlog_before_by_claimant: backlogBeforeByClaimant,
      allocations, local_costs: localCosts,
    };
    return deepCopy(this.prepared);
  }

  observation(role, { episodeId, recentHistory, cumulativeLocalCost }) {
    if (this.prepared === null) throw new Error("prepareWeek() must be called before observation");
    const state = this.states[role];
    const incoming = Object.values(this.prepared.incoming_by_claimant[role])
      .reduce((sum, value) => sum + value, 0);
    const filled = Object.values(this.prepared.allocations[role])
      .reduce((sum, value) => sum + value, 0);
    const stateView = {
      inventory_on_hand: state.inventory,
      backlog: state.backlog,
      inventory_position: this.inventoryPosition(role),
      on_order: this.onOrder(role),
      shipment_received: this.prepared.received[role],
      incoming_demand_or_order: incoming,
      units_filled: filled,
      last_order_placed: state.last_order_placed,
    };
    if (this.spec.observation_mode === "shipment_notices") {
      stateView.inbound_shipment_pipeline = [...state.shipment_pipeline];
    }
    return {
      episode_id: episodeId,
      scenario_id: this.spec.scenario_id,
      week: this.prepared.week,
      horizon: this.spec.horizon,
      weeks_remaining: this.spec.horizon - this.prepared.week + 1,
      role,
      topology: this.spec.topology,
      observation_mode: this.spec.observation_mode,
      state: stateView,
      costs: {
        holding_per_unit: this.spec.holding_cost,
        backlog_per_unit: this.spec.backlog_cost,
        current_inventory_backlog_cost: this.prepared.local_costs[role],
        cumulative_local_cost_through_previous_week: cumulativeLocalCost,
      },
      constraints: {
        minimum_order: 0,
        maximum_order: this.spec.order_cap,
        factory_capacity: this.spec.capacity,
      },
      recent_history: deepCopy(recentHistory.slice(-this.spec.history_window)),
    };
  }

  commitOrders(orders) {
    if (this.prepared === null) throw new Error("prepareWeek() must be called before commitOrders()");
    const keys = Object.keys(orders);
    const missing = this.spec.roles.filter((role) => !keys.includes(role));
    const extra = keys.filter((role) => !this.spec.roles.includes(role));
    if (missing.length || extra.length) {
      throw new RangeError(`orders must cover every role; missing=${missing}, extra=${extra}`);
    }
    for (const [role, quantity] of Object.entries(orders)) {
      if (!Number.isInteger(quantity) || quantity < 0 || quantity > this.spec.order_cap) {
        throw new RangeError(`invalid order for ${role}: ${quantity}`);
      }
    }

    const prepared = this.prepared;
    const statesBeforeOrders = Object.fromEntries(
      this.spec.roles.map((role) => [role, stateSnapshot(this.states[role])]),
    );
    let capacityBound = false;
    let production = 0;
    for (const role of this.spec.roles) {
      const quantity = orders[role];
      const state = this.states[role];
      state.last_order_placed = quantity;
      const upstream = this.upstream[role];
      if (upstream === null) {
        production = quantity;
        if (this.spec.capacity !== null && production > this.spec.capacity) {
          production = this.spec.capacity;
          capacityBound = true;
        }
        state.shipment_pipeline.push(production);
      } else {
        this.states[upstream].order_pipelines[role].push(quantity);
      }
    }
    const transition = {
      week: prepared.week,
      operational: prepared.operational,
      received: { ...prepared.received },
      incoming_by_claimant: deepCopy(prepared.incoming_by_claimant),
      backlog_before_by_claimant: deepCopy(prepared.backlog_before_by_claimant),
      allocations: deepCopy(prepared.allocations),
      states_after_fulfillment: statesBeforeOrders,
      orders: { ...orders },
      factory_production: production,
      capacity_bound: capacityBound,
      local_costs: { ...prepared.local_costs },
      system_cost: Object.values(prepared.local_costs).reduce((sum, value) => sum + value, 0),
    };
    this.week += 1;
    this.prepared = null;
    return transition;
  }

  snapshot() {
    return {
      week: this.week,
      scenario: deepCopy(this.spec),
      states: Object.fromEntries(this.spec.roles.map((role) => [role, stateSnapshot(this.states[role])])),
    };
  }

  canonicalSnapshot() {
    return stableStringify(this.snapshot());
  }
}
