import { firstUint64, sha256Bytes, sha256Hex } from "./sha256.js";

export const SERIAL_ROLES = ["retailer", "wholesaler", "distributor", "factory"];
export const Y_ROLES = ["retailer_a", "retailer_b", "wholesaler", "distributor", "factory"];
export const SPLIT_SIZES = { development: 3, validation: 5, test: 10 };
export const PLAYABLE_SEEDS = [
  ...Array.from({ length: 3 }, (_, seedIndex) => ({ split: "development", seedIndex })),
  ...Array.from({ length: 5 }, (_, seedIndex) => ({ split: "validation", seedIndex })),
];

function assertSplit(split, seedIndex) {
  if (!(split in SPLIT_SIZES)) throw new RangeError(`unknown split ${split}`);
  if (!Number.isInteger(seedIndex) || seedIndex < 0 || seedIndex >= SPLIT_SIZES[split]) {
    throw new RangeError(`seed index ${seedIndex} is outside '${split}' split`);
  }
}

export function masterSeedHex(split, seedIndex) {
  assertSplit(split, seedIndex);
  return sha256Hex(`beer-agent-v1|${split}|${String(seedIndex).padStart(5, "0")}`).slice(0, 16);
}

export function deriveSeed(seedHex, namespace) {
  return firstUint64(`beer-agent-v1|${seedHex}|${namespace}`);
}

function shock(seedHex) {
  const digest = sha256Bytes(`beer-agent-v1|${seedHex}|mechanism/shock`);
  return {
    shiftWeek: [15, 19, 23][digest[0] % 3],
    muAfter: [4.0, 12.0][digest[1] % 2],
  };
}

function commonScenario(tier, split, seedIndex) {
  return {
    schema_version: "1.0.0",
    environment_version: "0.2.0",
    tier,
    variant: "headline",
    split,
    seed_index: seedIndex,
    master_seed_hex: masterSeedHex(split, seedIndex),
    horizon: 36,
    order_delay: 1,
    shipment_delay: 2,
    order_cap: 128,
    holding_cost: 0.5,
    backlog_cost: 1.0,
    initial_inventory: 12,
    initial_shipment_pipeline: 4,
    initial_order_pipeline: 4,
    history_window: 8,
  };
}

export function scenarioFor(tier, split, seedIndex, variant = "headline") {
  if (!Number.isInteger(tier) || tier < 1 || tier > 5) throw new RangeError("tier must be in 1..5");
  if (tier !== 5 && variant !== "headline") throw new RangeError("Tier 5 controls require tier 5");
  const common = commonScenario(tier, split, seedIndex);
  common.variant = variant;

  if (tier === 1) {
    return {
      ...common, scenario_id: "t1-steady-serial-v2", topology: "serial",
      roles: [...SERIAL_ROLES], observation_mode: "shipment_notices",
      demand_process: "constant_v1", demand_parameters: { value: 8 },
      capacity: null, rationing: "proportional",
      counterparty_policy: "adaptive_base_stock_v2", aggressive_retailers: false,
    };
  }
  if (tier === 2) {
    return {
      ...common, scenario_id: "t2-ar1-serial-v2", topology: "serial",
      roles: [...SERIAL_ROLES], observation_mode: "shipment_notices",
      demand_process: "ar1_v1",
      demand_parameters: { mu: 7.5, phi: 0.7, sigma: 2.0, x0: 7.5 },
      capacity: null, rationing: "proportional",
      counterparty_policy: "adaptive_base_stock_v2", aggressive_retailers: false,
    };
  }
  if (tier === 3 || tier === 4) {
    const { shiftWeek, muAfter } = shock(common.master_seed_hex);
    return {
      ...common,
      scenario_id: tier === 3 ? "t3-shift-serial-v2" : "t4-partial-shift-serial-v2",
      topology: "serial", roles: [...SERIAL_ROLES],
      observation_mode: tier === 3 ? "shipment_notices" : "aggregate_supply_line",
      demand_process: "shifted_ar1_v1",
      demand_parameters: {
        mu_before: 7.5, phi: 0.7, sigma: 2.0, x0: 7.5,
        shift_week: shiftWeek, mu_after: muAfter,
      },
      capacity: null, rationing: "proportional",
      counterparty_policy: "adaptive_base_stock_v2", aggressive_retailers: false,
    };
  }

  const aggressive = variant !== "t5_control_base_rival";
  const rationing = variant === "t5_control_uniform" ? "uniform" : "proportional";
  const suffix = {
    headline: "strategic",
    t5_control_base_rival: "control-base-rival",
    t5_control_uniform: "control-uniform",
  }[variant];
  if (!suffix) throw new RangeError(`unknown variant ${variant}`);
  return {
    ...common, scenario_id: `t5-${suffix}-y-v2`, topology: "y", roles: [...Y_ROLES],
    observation_mode: "aggregate_supply_line",
    demand_process: "correlated_y_ar1_v1",
    demand_parameters: {
      mu: 7.5, phi: 0.7, sigma_common: 2.0,
      sigma_idiosyncratic: 1.5, common0: 0.0,
    },
    capacity: 22, rationing,
    counterparty_policy: "adaptive_base_stock_v2", aggressive_retailers: aggressive,
  };
}

// Python's json module retains .0 for values created as floats. These paths are
// the only integral floats in ScenarioSpec and therefore affect episode_id.
const FLOAT_PATHS = new Set([
  "backlog_cost", "holding_cost",
  "demand_parameters.common0", "demand_parameters.mu",
  "demand_parameters.mu_after", "demand_parameters.mu_before",
  "demand_parameters.phi", "demand_parameters.sigma",
  "demand_parameters.sigma_common", "demand_parameters.sigma_idiosyncratic",
  "demand_parameters.x0",
]);

function scenarioJson(value, path = "") {
  if (value === null) return "null";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new TypeError("non-finite scenario value");
    if (Number.isInteger(value) && FLOAT_PATHS.has(path)) return `${value}.0`;
    return String(value);
  }
  if (typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((item) => scenarioJson(item, path)).join(",")}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map((key) => {
    const childPath = path ? `${path}.${key}` : key;
    return `${JSON.stringify(key)}:${scenarioJson(value[key], childPath)}`;
  }).join(",")}}`;
}

export function canonicalScenarioJson(spec) {
  return scenarioJson(spec);
}

export function episodeId(spec, controlledRole) {
  return `sha256:${sha256Hex(`${canonicalScenarioJson(spec)}|${controlledRole}`)}`;
}

export function settlementWeeks(spec) {
  return spec.order_delay + spec.shipment_delay;
}
