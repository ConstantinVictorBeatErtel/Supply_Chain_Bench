function sumRoleCost(transitions, role) {
  return transitions.reduce((sum, row) => sum + Number(row.local_costs[role]), 0);
}

function terminalCost(spec, position) {
  return position >= 0 ? spec.holding_cost * position : spec.backlog_cost * -position;
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function variance(values) {
  if (values.length < 2) return null;
  // statistics.variance() evaluates integer inputs through an exact Fraction
  // path before converting to binary64. Preserve that ordering for parity.
  if (values.every(Number.isInteger)) {
    const n = BigInt(values.length);
    const sum = values.reduce((total, value) => total + BigInt(value), 0n);
    const squares = values.reduce(
      (total, value) => total + BigInt(value) * BigInt(value),
      0n,
    );
    return Number(n * squares - sum * sum) / Number(n * (n - 1n));
  }
  const average = mean(values);
  return values.reduce((sumSquares, value) => sumSquares + (value - average) ** 2, 0)
    / (values.length - 1);
}

export function serviceMetrics(transitions, role) {
  let totalNew = 0;
  let totalImmediate = 0;
  let positiveCycles = 0;
  let successfulCycles = 0;
  for (const row of transitions) {
    const incoming = row.incoming_by_claimant[role];
    const backlogBefore = row.backlog_before_by_claimant[role];
    const allocation = row.allocations[role];
    for (const [claimant, rawDemand] of Object.entries(incoming)) {
      const demand = Number(rawDemand);
      const prior = Number(backlogBefore[claimant] || 0);
      const shipped = Number(allocation[claimant] || 0);
      const immediate = Math.min(demand, Math.max(0, shipped - prior));
      totalNew += demand;
      totalImmediate += immediate;
      if (demand > 0) {
        positiveCycles += 1;
        successfulCycles += Number(immediate === demand);
      }
    }
  }
  const endingBacklog = transitions.at(-1).states_after_fulfillment[role].backlog;
  return {
    immediate_fill_rate: totalNew ? totalImmediate / totalNew : null,
    cycle_service_level: positiveCycles ? successfulCycles / positiveCycles : null,
    horizon_fulfillment: totalNew ? (totalNew - endingBacklog) / totalNew : null,
    ending_backlog: endingBacklog,
    new_demand_units: totalNew,
  };
}

export function stabilityMetrics(spec, transitions, role) {
  const warmup = spec.order_delay + spec.shipment_delay;
  const scored = transitions.slice(warmup);
  const orders = scored.map((row) => Number(row.orders[role]));
  const demands = scored.map((row) => Object.values(row.incoming_by_claimant[role])
    .reduce((sum, value) => sum + Number(value), 0));
  const demandVariance = variance(demands);
  const orderVariance = variance(orders);
  const bullwhip = demandVariance !== null && demandVariance > 1e-12 && orderVariance !== null
    ? orderVariance / demandVariance
    : null;
  const volatility = orders.length >= 2
    ? mean(orders.slice(1).map((value, index) => Math.abs(value - orders[index])))
      / Math.max(mean(demands), 1)
    : null;
  const totalDemand = demands.reduce((sum, value) => sum + value, 0);
  return {
    metric_warmup_weeks: warmup,
    bullwhip_ratio: bullwhip,
    normalized_order_volatility: volatility,
    order_mean: orders.length ? mean(orders) : null,
    order_median: orders.length ? median(orders) : null,
    order_variance: orderVariance,
    demand_variance: demandVariance,
    maximum_order: orders.length ? Math.max(...orders) : null,
    order_cap_hit_rate: orders.length
      ? orders.filter((order) => order === spec.order_cap).length / orders.length
      : null,
    order_to_demand_ratio: totalDemand
      ? orders.reduce((sum, value) => sum + value, 0) / totalDemand
      : null,
  };
}

export function gradeEpisode({
  spec, controlledRole, operational, settlement,
  terminalInventoryPositions, protocolClean, baseReference = null,
}) {
  const operationalLocal = sumRoleCost(operational, controlledRole);
  const settlementLocal = sumRoleCost(settlement, controlledRole);
  const terminalByRole = Object.fromEntries(
    spec.roles.map((role) => [role, terminalCost(spec, terminalInventoryPositions[role])]),
  );
  const terminalLocal = terminalByRole[controlledRole];
  const localTotal = operationalLocal + settlementLocal + terminalLocal;
  const operationalSystem = spec.roles.reduce((sum, role) => sum + sumRoleCost(operational, role), 0);
  const settlementSystem = spec.roles.reduce((sum, role) => sum + sumRoleCost(settlement, role), 0);
  const terminalSystem = Object.values(terminalByRole).reduce((sum, value) => sum + value, 0);
  const systemTotal = operationalSystem + settlementSystem + terminalSystem;
  const baseLocal = baseReference === null ? null : baseReference.local_total_cost;
  const baseSystem = baseReference === null ? null : baseReference.system_total_cost;
  let costScore = null;
  if (baseLocal !== null) {
    costScore = baseLocal > 0 ? baseLocal / (baseLocal + localTotal) : Number(localTotal === 0);
  }
  return {
    grader_version: "1.0.0",
    status: costScore === null ? "reference" : "scored",
    episode_reward: costScore === null ? null : Number(protocolClean) * costScore,
    protocol_clean: protocolClean,
    primary: {
      local_total_cost: localTotal,
      paired_base_stock_local_total_cost: baseLocal,
      cost_score: costScore,
    },
    costs: {
      operational_local_cost: operationalLocal,
      settlement_local_cost: settlementLocal,
      terminal_exposure_cost: terminalLocal,
      mean_weekly_local_cost: operationalLocal / spec.horizon,
      system_total_cost: systemTotal,
      operational_system_cost: operationalSystem,
      settlement_system_cost: settlementSystem,
      terminal_system_exposure_cost: terminalSystem,
      other_roles_total_cost: systemTotal - localTotal,
      paired_base_stock_system_total_cost: baseSystem,
      local_cost_ratio: baseLocal !== null && baseLocal !== 0 ? localTotal / baseLocal : null,
      system_cost_ratio: baseSystem !== null && baseSystem !== 0 ? systemTotal / baseSystem : null,
    },
    service: serviceMetrics(operational, controlledRole),
    stability: stabilityMetrics(spec, operational, controlledRole),
    termination_reason: "horizon_completed",
  };
}
