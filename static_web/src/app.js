import { BeerEpisode, replayActions, scenarioFor } from "./sim/index.js";
import {
  configureTelemetry, createSessionUuid, sendTelemetry,
} from "./telemetry.js";

const TIER = 5;
const VARIANT = "headline";
const ROLES = ["retailer_a", "retailer_b", "wholesaler", "distributor", "factory"];
const ROLE_LABEL = {
  retailer_a: "Retailer A", retailer_b: "Retailer B", wholesaler: "Wholesaler",
  distributor: "Distributor", factory: "Factory",
};
const ROLE_NOTE = {
  retailer_a: "Faces end customers directly. Shortest signal, sharpest noise.",
  retailer_b: "Competes with A for the same wholesaler inventory.",
  wholesaler: "Rations one pool between two retailers. The hard seat.",
  distributor: "Buffers the factory against wholesaler swings.",
  factory: "Produces to order, capped at 22 units a week.",
};

let catalog = null;
let active = null;
let selectedRole = "wholesaler";
let logSent = false;
let fastForwardTimer = null;

export function parseOrderInput(raw) {
  if (typeof raw !== "string" || !/^(0|[1-9]\d{0,2})$/.test(raw)) {
    return { ok: false, message: "Enter one whole number from 0 through 128." };
  }
  const quantity = Number(raw);
  if (!Number.isInteger(quantity) || quantity < 0 || quantity > 128) {
    return { ok: false, message: "Enter one whole number from 0 through 128." };
  }
  return { ok: true, quantity };
}

function escapeHtml(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function n0(value) { return Math.round(Number(value) || 0).toLocaleString("en-US"); }
function n1(value) {
  return (Math.round((Number(value) || 0) * 10) / 10).toLocaleString("en-US", {
    minimumFractionDigits: 1, maximumFractionDigits: 1,
  });
}

function scenarioForTrace(trace) {
  const split = trace.seed_set === "live_y_research_eval" ? "test" : trace.split;
  const spec = scenarioFor(TIER, split, trace.seed_index);
  if (trace.seed_set === "live_y_research_eval") spec.split = "research_eval";
  spec.master_seed_hex = trace.master_seed_hex;
  return spec;
}

function setHeader(week = "—", cost = "—") {
  const header = document.querySelector("#header-status");
  if (header) header.innerHTML = `<span><b>SEAT</b>${escapeHtml(ROLE_LABEL[selectedRole].toUpperCase())}</span><span><b>WEEK</b>${week}</span><span><b>COST</b>${cost}</span>`;
}

function lineChart(values, max, width = 720, height = 190) {
  if (!values.length) return "";
  const top = Math.max(1, max);
  return values.map((value, index) => {
    const x = values.length === 1 ? 0 : (index / 35) * width;
    const y = height - Math.max(0, Math.min(1, value / top)) * (height - 6) - 3;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

function chainHtml({ briefing = false } = {}) {
  const role = active?.role || selectedRole;
  const transition = active?.episode.operationalTransitions.at(-1);
  const states = transition?.states_after_fulfillment || {};
  const cards = ROLES.map((node) => {
    const isYou = node === role;
    const state = states[node];
    const visible = !briefing && isYou && state;
    const detail = briefing
      ? `<p>${escapeHtml(ROLE_NOTE[node])}</p>`
      : `<dl><div><dt>INV</dt><dd>${visible ? n0(state.inventory) : "——"}</dd></div><div><dt>BL</dt><dd class="${visible && state.backlog > 0 ? "danger" : ""}">${visible ? n0(state.backlog) : "——"}</dd></div></dl>`;
    return `<button class="chain-card chain-${node} ${isYou ? "selected" : ""}" data-role="${node}" type="button" ${briefing ? "" : "disabled"}>
      <span class="chain-tag">${isYou ? "YOU" : briefing ? "AVAILABLE" : "SEALED"}</span>
      <strong>${ROLE_LABEL[node]}</strong>${detail}</button>`;
  });
  return `<div class="chain" aria-label="Five-node Y supply chain">
    <div class="retailers">${cards[0]}${cards[1]}</div>
    <div class="fork" aria-hidden="true"><i></i><i></i><b></b></div>
    ${cards[2]}<span class="link link-one" aria-hidden="true"></span>${cards[3]}<span class="link link-two" aria-hidden="true"></span>${cards[4]}
  </div>`;
}

function briefingHtml() {
  return `<section class="briefing" aria-labelledby="briefing-title">
    <div class="hero">
      <h1 id="briefing-title">You hold one seat in a five-node supply chain.</h1>
      <p>Every week you see your own inventory, your own backlog, and the order that arrived from downstream. Nothing else. You place one order upstream and it lands three weeks later. Thirty-six weeks. Holding costs 0.5 per unit per week, backlog costs 1.0.</p>
      <p>A language model plays the same seat, on the same seed, against the same counterparties, in a separate sealed episode. Neither of you sees the other until week 36.</p>
    </div>
    <p class="section-label">Choose your seat</p>
    ${chainHtml({ briefing: true })}
    <div class="start-row"><button id="start-game" class="primary-button" type="button">Begin week 1 <span>→</span></button></div>
  </section>`;
}

function statHtml(label, value, danger = false) {
  return `<div class="stat"><span>${escapeHtml(label)}</span><strong class="${danger ? "danger" : ""}">${escapeHtml(value)}</strong></div>`;
}

function graphHtml(title, labels, lines, max) {
  return `<section class="graph">
    <div class="graph-head"><span>${escapeHtml(title)}</span><div>${labels.map((label) => `<em class="${label.className}">— ${escapeHtml(label.text)}</em>`).join("")}</div></div>
    <div class="chart"><svg viewBox="0 0 720 190" preserveAspectRatio="none" role="img" aria-label="${escapeHtml(title)}">
      <line x1="0" y1="95" x2="720" y2="95"></line><line class="axis" x1="0" y1="190" x2="720" y2="190"></line>
      ${lines.map((line) => `<polyline class="${line.className}" points="${lineChart(line.values, max)}"></polyline>`).join("")}
    </svg><small>${n0(max)}</small></div>
    <div class="chart-weeks"><span>WK 1</span><span>WK 36</span></div>
  </section>`;
}

function gameHtml() {
  const { observation, episode, order } = active;
  const { state, costs } = observation;
  const history = episode.histories[active.role];
  const inventory = history.map((row) => row.ending_inventory);
  const backlog = history.map((row) => row.ending_backlog);
  const orders = history.map((row) => row.order_placed);
  const incoming = history.map((row) => row.incoming_demand_or_order);
  const stateMax = Math.max(8, ...inventory, ...backlog);
  const orderMax = Math.max(8, ...orders, ...incoming);
  return `<section class="game" aria-labelledby="position-title">
    ${chainHtml()}
    <div class="game-grid">
      <div class="decision-panel">
        <section><p id="position-title" class="section-label">Your position · week ${observation.week}</p>
          <div class="stats-grid">
            ${statHtml("Inventory on hand", n0(state.inventory_on_hand))}
            ${statHtml("Backlog", n0(state.backlog), state.backlog > 0)}
            ${statHtml("Inventory position", n0(state.inventory_position))}
            ${statHtml("On order", n0(state.on_order))}
            ${statHtml("Shipment received", n0(state.shipment_received))}
            ${statHtml("Demand on you", n0(state.incoming_demand_or_order))}
            ${statHtml("Units filled", n0(state.units_filled))}
            ${statHtml("Week cost", n1(costs.current_inventory_backlog_cost))}
          </div>
        </section>
        <form id="order-form" class="order-form" novalidate>
          <p class="section-label">Place order upstream</p>
          <div class="order-number"><output id="order-value">${order}</output><span>units · arrives wk ${Math.min(39, observation.week + 3)}</span></div>
          <input id="order-range" type="range" min="0" max="128" step="1" value="${order}" aria-label="Order quantity">
          <div class="range-labels"><span>0</span><span>128</span></div>
          <p id="order-error" class="error-message" role="alert"></p>
          <div class="order-actions"><button id="minus-order" type="button" aria-label="Decrease order">−</button><button id="plus-order" type="button" aria-label="Increase order">+</button><button class="primary-button" type="submit">Advance week</button></div>
          <button id="fast-forward" class="hold-button" type="button">Hold to repeat ▸▸</button>
        </form>
      </div>
      <div class="graphs">
        ${graphHtml("Inventory and backlog", [{ text: "On hand", className: "blue" }, { text: "Backlog", className: "red" }], [{ values: inventory, className: "blue" }, { values: backlog, className: "red" }], stateMax)}
        ${graphHtml("Orders out vs demand in", [{ text: "You ordered", className: "light" }, { text: "Demand on you", className: "muted" }], [{ values: incoming, className: "muted" }, { values: orders, className: "light" }], orderMax)}
        <div class="scorecard">${statHtml("Cost through prior week", n1(costs.cumulative_local_cost_through_previous_week))}${statHtml("Weeks remaining", observation.weeks_remaining)}${statHtml("Holding / backlog", "0.5 / 1.0")}${statHtml("Factory capacity", "22")}</div>
      </div>
    </div>
  </section>`;
}

function ratio(values) {
  if (values.length < 2) return "—";
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1);
  return n1(variance);
}

function debriefHtml() {
  const grade = active.episode.outcome.grade;
  const humanCost = grade.primary.local_total_cost;
  const baseCost = grade.primary.paired_base_stock_local_total_cost;
  let modelCost = null;
  if (active.role === "wholesaler" && active.trace) {
    const replay = replayActions(scenarioForTrace(active.seed), active.role, active.trace.actions).episode;
    modelCost = replay.outcome.grade.primary.local_total_cost;
  }
  const lead = modelCost === null ? "Your 36-week episode is complete." : humanCost <= modelCost ? "You beat the recorded model." : "The recorded model finished lower.";
  const rows = ROLES.map((role) => {
    const history = active.episode.histories[role];
    const orders = history.map((row) => row.order_placed);
    const demand = history.map((row) => row.incoming_demand_or_order);
    const avg = orders.reduce((sum, value) => sum + value, 0) / Math.max(1, orders.length);
    return `<div class="chain-row ${role === active.role ? "you" : ""}"><span>${ROLE_LABEL[role]}</span><span>${n1(avg)}</span><span>${n0(Math.max(0, ...orders))}</span><span>${ratio(orders)}/${ratio(demand)}</span><span>${n1(active.episode.cumulativeCosts[role])}</span></div>`;
  }).join("");
  const yours = active.episode.histories[active.role].map((row) => row.order_placed);
  const demand = active.episode.histories[active.role].map((row) => row.incoming_demand_or_order);
  const comparison = modelCost === null ? [] : active.trace.actions;
  const max = Math.max(8, ...yours, ...demand, ...comparison);
  return `<section class="debrief" aria-labelledby="debrief-title">
    <div class="hero"><h1 id="debrief-title">${lead}</h1><p>Totals include the 36 operational weeks, deterministic settlement, and terminal inventory-position exposure.</p></div>
    <div class="final-cards">
      <article><span>YOUR COST</span><strong>${n1(humanCost)}</strong><p>Local total cost at ${ROLE_LABEL[active.role]}.</p></article>
      <article><span>${modelCost === null ? "REFERENCE COST" : "RECORDED MODEL"}</span><strong>${n1(modelCost ?? baseCost)}</strong><p>${modelCost === null ? "Adaptive base-stock comparison." : "Same seat and same deterministic seed."}</p></article>
      <article><span>BASE-STOCK</span><strong>${n1(baseCost)}</strong><p>Adaptive policy reference · score ${(grade.episode_reward * 100).toFixed(1)}.</p></article>
    </div>
    ${graphHtml("Orders placed · fog lifted", [{ text: "You", className: "light" }, { text: modelCost === null ? "Reference unavailable" : "Recorded model", className: "blue" }, { text: "Demand", className: "muted" }], [{ values: demand, className: "muted" }, { values: comparison, className: "blue" }, { values: yours, className: "light" }], max)}
    <section class="chain-table"><p class="section-label">Full chain · your episode</p><div class="chain-row heading"><span>Node</span><span>Mean order</span><span>Peak order</span><span>Order/demand variance</span><span>Operational cost</span></div>${rows}</section>
    <div class="start-row"><button id="new-game" class="primary-button" type="button">Play again</button></div>
  </section>`;
}

function recordFor(status) {
  const completed = status === "completed";
  const grade = completed ? active.episode.outcome.grade : null;
  return {
    session_uuid: active.sessionUuid, timestamp: active.timestamp, env_version: "0.2.0",
    tier: TIER, role: active.role, split: active.episode.spec.split,
    seed_index: active.seed.seed_index, seed: active.episode.spec.master_seed_hex,
    scenario_id: active.episode.spec.scenario_id, variant: VARIANT, status, completed,
    actions: [...active.actions], weekly: [...active.weekly],
    final_total_cost: completed ? grade.primary.local_total_cost : null,
    base_stock_cost: completed ? grade.primary.paired_base_stock_local_total_cost : null,
    episode_reward: completed ? grade.episode_reward : null,
  };
}

function sendCurrentRecord(status) {
  if (!active || logSent) return false;
  logSent = true;
  return sendTelemetry(recordFor(status));
}

function bindBriefing() {
  document.querySelectorAll("[data-role]").forEach((card) => card.addEventListener("click", () => {
    selectedRole = card.dataset.role;
    renderBriefing();
  }));
  document.querySelector("#start-game").addEventListener("click", () => {
    const random = new Uint32Array(1);
    globalThis.crypto?.getRandomValues?.(random);
    const catalogIndex = random[0] % catalog.seeds.length;
    const seed = catalog.seeds[catalogIndex];
    const spec = selectedRole === "wholesaler" ? scenarioForTrace(seed) : scenarioFor(TIER, "development", catalogIndex % 3);
    const episode = new BeerEpisode(spec, selectedRole);
    active = { seed, episode, role: selectedRole, observation: episode.start(), order: 8, actions: [], weekly: [], sessionUuid: createSessionUuid(), timestamp: new Date().toISOString(), trace: selectedRole === "wholesaler" ? seed : null };
    logSent = false;
    renderGame();
  });
}

function renderBriefing() {
  clearInterval(fastForwardTimer);
  active = null;
  logSent = false;
  document.querySelector("#app").innerHTML = briefingHtml();
  setHeader();
  bindBriefing();
}

function updateOrder(value) {
  active.order = Math.max(0, Math.min(128, Math.round(value)));
  document.querySelector("#order-range").value = String(active.order);
  document.querySelector("#order-value").textContent = String(active.order);
}

function commitOrder() {
  if (!active || active.episode.done) return;
  const result = active.episode.placeOrder(active.order);
  active.actions.push(active.order);
  const transition = active.episode.operationalTransitions.at(-1);
  const state = transition.states_after_fulfillment[active.role];
  active.weekly.push({ week: transition.week, inventory: state.inventory, backlog: state.backlog, local_cost: transition.local_costs[active.role] });
  if (result.done) {
    clearInterval(fastForwardTimer);
    sendCurrentRecord("completed");
    document.querySelector("#app").innerHTML = debriefHtml();
    setHeader("36/36", n1(active.episode.cumulativeCosts[active.role]));
    document.querySelector("#new-game").addEventListener("click", renderBriefing);
    return;
  }
  active.observation = result.next_observation;
  renderGame();
}

function renderGame() {
  document.querySelector("#app").innerHTML = gameHtml();
  setHeader(`${active.observation.week}/36`, n1(active.observation.costs.cumulative_local_cost_through_previous_week));
  const range = document.querySelector("#order-range");
  range.addEventListener("input", () => updateOrder(Number(range.value)));
  document.querySelector("#minus-order").addEventListener("click", () => updateOrder(active.order - 1));
  document.querySelector("#plus-order").addEventListener("click", () => updateOrder(active.order + 1));
  document.querySelector("#order-form").addEventListener("submit", (event) => { event.preventDefault(); commitOrder(); });
  const hold = document.querySelector("#fast-forward");
  const stop = () => clearInterval(fastForwardTimer);
  hold.addEventListener("pointerdown", () => { commitOrder(); fastForwardTimer = setInterval(commitOrder, 220); });
  ["pointerup", "pointerleave", "pointercancel"].forEach((event) => hold.addEventListener(event, stop));
}

async function loadCatalog() {
  const response = await fetch("./data/llm-comparison.json", { cache: "no-cache" });
  if (!response.ok) throw new Error(`trace catalog returned ${response.status}`);
  const payload = await response.json();
  if (!Array.isArray(payload.seeds) || ![8, 10].includes(payload.seeds.length)) throw new Error("trace catalog must contain replayable model traces");
  return payload;
}

export async function initialize() {
  configureTelemetry(globalThis.BEER_GAME_CONFIG?.loggingEndpoint || "");
  try { catalog = await loadCatalog(); renderBriefing(); } catch (error) {
    console.error(error);
    document.querySelector("#app").innerHTML = '<section class="loading-card"><p class="section-label">Unable to start</p><h1>Scenario data could not be loaded.</h1><p>Reload the page to try again.</p></section>';
  }
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  window.addEventListener("pagehide", () => { if (active && !active.episode.done) sendCurrentRecord("abandoned"); });
  window.addEventListener("keydown", (event) => {
    if (!active || active.episode.done) return;
    if (event.key === "Enter") commitOrder();
    if (event.key === "ArrowUp") { event.preventDefault(); updateOrder(active.order + 1); }
    if (event.key === "ArrowDown") { event.preventDefault(); updateOrder(active.order - 1); }
  });
  void initialize();
}
