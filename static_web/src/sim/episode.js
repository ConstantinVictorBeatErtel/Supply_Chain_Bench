import { deepCopy } from "./canonical.js";
import { BeerGameCore } from "./core.js";
import { gradeEpisode } from "./grading.js";
import { adaptivePolicy, counterpartyPolicies } from "./policies.js";
import { episodeId, settlementWeeks } from "./scenario.js";

export class BeerEpisode {
  constructor(spec, controlledRole, { includeReference = true } = {}) {
    if (!spec.roles.includes(controlledRole)) throw new RangeError(`role '${controlledRole}' is unavailable`);
    this.spec = deepCopy(spec);
    this.controlledRole = controlledRole;
    this.episodeId = episodeId(this.spec, controlledRole);
    this.includeReference = includeReference;
    this.core = new BeerGameCore(this.spec);
    this.counterparties = counterpartyPolicies(this.spec, controlledRole);
    this.histories = Object.fromEntries(this.spec.roles.map((role) => [role, []]));
    this.cumulativeCosts = Object.fromEntries(this.spec.roles.map((role) => [role, 0]));
    this.operationalTransitions = [];
    this.settlementTransitions = [];
    this.protocolClean = true;
    this.started = false;
    this.done = false;
    this.outcome = null;
  }

  observations() {
    return Object.fromEntries(this.spec.roles.map((role) => [
      role,
      this.core.observation(role, {
        episodeId: this.episodeId,
        recentHistory: this.histories[role],
        cumulativeLocalCost: this.cumulativeCosts[role],
      }),
    ]));
  }

  start() {
    if (this.started) throw new Error("episode already started");
    this.started = true;
    this.core.prepareWeek({ operational: true });
    return this.observations()[this.controlledRole];
  }

  placeOrder(quantity) {
    if (!this.started || this.done) throw new Error("order is out of turn");
    if (!Number.isInteger(quantity) || quantity < 0 || quantity > this.spec.order_cap) {
      throw new RangeError(`quantity must be an integer in [0, ${this.spec.order_cap}]`);
    }
    const observations = this.observations();
    const orders = { [this.controlledRole]: quantity };
    for (const [role, policy] of Object.entries(this.counterparties)) {
      orders[role] = policy.act(observations[role]);
    }
    const transition = this.core.commitOrders(orders);
    this.operationalTransitions.push(transition);
    for (const role of this.spec.roles) {
      const localCost = Number(transition.local_costs[role]);
      const state = transition.states_after_fulfillment[role];
      this.histories[role].push({
        week: transition.week,
        incoming_demand_or_order: Object.values(transition.incoming_by_claimant[role])
          .reduce((sum, value) => sum + value, 0),
        shipment_received: transition.received[role],
        order_placed: transition.orders[role],
        ending_inventory: state.inventory,
        ending_backlog: state.backlog,
        local_cost: localCost,
      });
      this.cumulativeCosts[role] += localCost;
    }
    if (this.operationalTransitions.length >= this.spec.horizon) {
      this.finish();
      return {
        status: "accepted", completed_week: transition.week, order_placed: quantity,
        done: true, termination_reason: "horizon_completed",
        summary: {
          episode_reward: this.outcome.grade.episode_reward,
          local_total_cost: this.outcome.grade.primary.local_total_cost,
          system_total_cost: this.outcome.grade.costs.system_total_cost,
        },
      };
    }
    this.core.prepareWeek({ operational: true });
    return {
      status: "accepted", completed_week: transition.week, order_placed: quantity,
      done: false, next_observation: this.observations()[this.controlledRole],
    };
  }

  finish() {
    for (let index = 0; index < settlementWeeks(this.spec); index += 1) {
      this.core.prepareWeek({ operational: false });
      this.settlementTransitions.push(this.core.commitOrders(
        Object.fromEntries(this.spec.roles.map((role) => [role, 0])),
      ));
    }
    const terminalPositions = Object.fromEntries(
      this.spec.roles.map((role) => [role, this.core.inventoryPosition(role)]),
    );
    const baseReference = this.includeReference ? runReference(this.spec, this.controlledRole) : null;
    const grade = gradeEpisode({
      spec: this.spec,
      controlledRole: this.controlledRole,
      operational: this.operationalTransitions,
      settlement: this.settlementTransitions,
      terminalInventoryPositions: terminalPositions,
      protocolClean: this.protocolClean,
      baseReference,
    });
    this.done = true;
    this.outcome = {
      episode_id: this.episodeId,
      scenario: deepCopy(this.spec),
      controlled_role: this.controlledRole,
      counterparties: Object.fromEntries(Object.entries(this.counterparties).map(([role, policy]) => [
        role, { policy_id: policy.policy_id, policy_version: policy.policy_version },
      ])),
      operational_transitions: this.operationalTransitions,
      settlement_transitions: this.settlementTransitions,
      terminal_inventory_positions: terminalPositions,
      final_state: this.core.snapshot(),
      grade,
    };
  }
}

export function runReference(spec, controlledRole) {
  const episode = new BeerEpisode(spec, controlledRole, { includeReference: false });
  const policy = adaptivePolicy(spec, controlledRole);
  let observation = episode.start();
  while (!episode.done) {
    const result = episode.placeOrder(policy.act(observation));
    if (!result.done) observation = result.next_observation;
  }
  return {
    local_total_cost: episode.outcome.grade.primary.local_total_cost,
    system_total_cost: episode.outcome.grade.costs.system_total_cost,
  };
}

export function replayActions(spec, controlledRole, actions, { includeReference = true } = {}) {
  const episode = new BeerEpisode(spec, controlledRole, { includeReference });
  let observation = episode.start();
  const observations = [observation];
  for (const quantity of actions) {
    const result = episode.placeOrder(quantity);
    if (!result.done) {
      observation = result.next_observation;
      observations.push(observation);
    }
  }
  return { episode, observations };
}
