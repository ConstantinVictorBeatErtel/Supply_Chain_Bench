import { BeerEpisode, replayActions, scenarioFor } from "./sim/index.js";
import {
  configureTelemetry, createSessionUuid, sendTelemetry,
} from "./telemetry.js";

const ROLE = "wholesaler";
const TIER = 5;
const VARIANT = "headline";

let catalog = null;
let active = null;
let logSent = false;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

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

function formatCost(value) {
  return Number(value).toLocaleString("en-US", {
    minimumFractionDigits: Number.isInteger(value) ? 0 : 1,
    maximumFractionDigits: 1,
  });
}

function setHeader(text) {
  const element = document.querySelector("#header-status");
  if (element) element.textContent = text;
}

function progressHtml(week) {
  return Array.from({ length: 36 }, (_, index) => {
    const number = index + 1;
    const className = number < week ? "complete" : number === week ? "current" : "";
    return `<span class="${className}"></span>`;
  }).join("");
}

function briefingHtml() {
  return `
    <section class="briefing" aria-labelledby="briefing-title">
      <div class="briefing-hero">
        <div>
          <p class="eyebrow">Matched human evaluation</p>
          <h1 id="briefing-title">Can you control the delay?</h1>
          <p class="lede">You control one local role in the same deterministic environment used for the recorded LLM evaluation. Make one replenishment decision per week and minimize only your own holding and backlog cost.</p>
        </div>
        <aside class="condition-card" aria-label="Fixed game condition">
          <dl>
            <div><dt>Tier</dt><dd>5 · Strategic scarcity</dd></div>
            <div><dt>Role</dt><dd>Wholesaler</dd></div>
            <div><dt>Topology</dt><dd>Y supply chain</dd></div>
            <div><dt>Horizon</dt><dd>36 weeks</dd></div>
          </dl>
        </aside>
      </div>
      <div class="briefing-grid">
        <section class="rules" aria-labelledby="rules-title">
          <p class="eyebrow">Task contract</p>
          <h2 id="rules-title">What is disclosed</h2>
          <ul class="rules-list">
            <li>Holding cost is 0.5 per ending-inventory unit each week; backlog cost is 1.0 per ending-backlog unit.</li>
            <li>Orders take 1 week and shipments take 2 weeks.</li>
            <li>Each retailer has persistent stochastic demand with long-run mean 7.5.</li>
            <li>Factory capacity is 22; shortages use proportional allocation.</li>
            <li>A scripted rival claimant may order aggressively. Its state and policy parameters are private.</li>
            <li>Place exactly one integer order from 0 through 128 each week.</li>
          </ul>
        </section>
        <form id="start-form" class="start-form">
          <p class="eyebrow">Session setup</p>
          <h2>Ready when you are?</h2>
          <p class="setup-copy">An approved evaluation seed will be assigned automatically. You will see the same local information available to the model, one week at a time.</p>
          <fieldset>
            <legend>Played the Beer Game before?</legend>
            <div class="radio-row">
              <label><input type="radio" name="experience" value="yes" required><span>Yes</span></label>
              <label><input type="radio" name="experience" value="no"><span>No</span></label>
              <label><input type="radio" name="experience" value="unsure"><span>Unsure</span></label>
            </div>
          </fieldset>
          <p class="notice">Anonymous telemetry is optional. No names or contact details are collected.</p>
          <button class="primary-button" type="submit">Begin 36-week game</button>
        </form>
      </div>
    </section>
  `;
}

function historyHtml(history) {
  if (!history.length) return '<p class="empty-history">No prior decisions yet.</p>';
  const rows = history.map((row) => `
    <tr>
      <td>${row.week}</td>
      <td>${row.incoming_demand_or_order}</td>
      <td>${row.shipment_received}</td>
      <td>${row.order_placed}</td>
      <td>${row.ending_inventory}</td>
      <td>${row.ending_backlog}</td>
      <td>${formatCost(row.local_cost)}</td>
    </tr>
  `).join("");
  return `
    <div class="table-wrap">
      <table>
        <thead><tr><th>Week</th><th>Inbound</th><th>Received</th><th>Ordered</th><th>End inv.</th><th>End backlog</th><th>Cost</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function metricCard(label, value, warning = false) {
  return `
    <div class="metric-card${warning ? " warn" : ""}">
      <span class="metric-label">${escapeHtml(label)}</span>
      <strong class="metric-value">${escapeHtml(value)}</strong>
    </div>
  `;
}

function chartPolyline(values, maxValue, width, height, padding) {
  const usableWidth = width - padding.left - padding.right;
  const usableHeight = height - padding.top - padding.bottom;
  if (!values.length) return "";
  return values.map((value, index) => {
    const x = values.length === 1
      ? padding.left + usableWidth / 2
      : padding.left + (index / (values.length - 1)) * usableWidth;
    const y = padding.top + usableHeight - (value / maxValue) * usableHeight;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

function trajectoryCharts(history) {
  if (!history.length) {
    return '<section class="charts-panel" aria-label="Local trajectory"><p class="empty-history">Your trajectory will appear here after the first decision.</p></section>';
  }
  const width = 640;
  const height = 190;
  const padding = { top: 18, right: 18, bottom: 28, left: 38 };
  const inventory = history.map((row) => row.inventory);
  const backlog = history.map((row) => row.backlog);
  const cumulative = [];
  history.reduce((total, row) => {
    const next = total + Number(row.local_cost);
    cumulative.push(next);
    return next;
  }, 0);
  const stateMax = Math.max(1, ...inventory, ...backlog);
  const costMax = Math.max(1, ...cumulative);
  const chart = (title, maxValue, series, labels, colors, valueLabel) => `
    <div class="chart-card">
      <div class="chart-heading">
        <h3>${title}</h3>
        <span>${valueLabel(maxValue)}</span>
      </div>
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${title}">
        <line class="chart-axis" x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" />
        <line class="chart-gridline" x1="${padding.left}" y1="${padding.top}" x2="${width - padding.right}" y2="${padding.top}" />
        ${series.map((values, index) => `<polyline class="chart-line" style="stroke:${colors[index]}" points="${chartPolyline(values, maxValue, width, height, padding)}" />`).join("")}
        <text class="chart-label" x="${padding.left}" y="${height - 8}">W1</text>
        <text class="chart-label" text-anchor="end" x="${width - padding.right}" y="${height - 8}">W${history.at(-1).week}</text>
        <text class="chart-label" x="8" y="${padding.top + 4}">${valueLabel(maxValue)}</text>
        <text class="chart-label" x="8" y="${height - padding.bottom + 4}">0</text>
      </svg>
      <div class="chart-legend">${labels.map((label, index) => `<span><i style="background:${colors[index]}"></i>${label}</span>`).join("")}</div>
    </div>
  `;
  return `
    <section class="charts-panel" aria-labelledby="trajectory-title">
      <div class="charts-heading">
        <div><p class="eyebrow">Your local trajectory</p><h2 id="trajectory-title">How the game is going</h2></div>
        <span class="chart-note">Observed weeks only</span>
      </div>
      <div class="charts-grid">
        ${chart("Inventory and backlog", stateMax, [inventory, backlog], ["Inventory", "Backlog"], ["#65d6a1", "#ff8c7d"], (value) => String(Math.round(value)))}
        ${chart("Cumulative local cost", costMax, [cumulative], ["Cost"], ["#f4b860"], (value) => formatCost(value))}
      </div>
    </section>
  `;
}

function gameHtml(observation) {
  const state = observation.state;
  const costs = observation.costs;
  return `
    <section class="game-panel" aria-labelledby="week-title">
      <div class="game-topbar">
        <div>
          <p class="eyebrow">Wholesaler · Tier 5 strategic</p>
          <p id="week-title" class="week-number">Week ${observation.week} / ${observation.horizon}</p>
        </div>
        <div class="progress-track" aria-label="${observation.week - 1} of 36 decisions complete">${progressHtml(observation.week)}</div>
      </div>
      <div class="game-layout">
        <div class="observation-column">
          <div class="state-grid">
            ${metricCard("Inventory on hand", state.inventory_on_hand)}
            ${metricCard("Backlog", state.backlog, state.backlog > 0)}
            ${metricCard("Inventory position", state.inventory_position)}
            ${metricCard("On order", state.on_order)}
            ${metricCard("Shipment received", state.shipment_received)}
            ${metricCard("Incoming order", state.incoming_demand_or_order)}
            ${metricCard("Units filled", state.units_filled)}
            ${metricCard("Last order placed", state.last_order_placed)}
            ${metricCard("Current local cost", formatCost(costs.current_inventory_backlog_cost))}
          </div>
          ${trajectoryCharts(active.weekly)}
          <section class="history-panel" aria-labelledby="history-title">
            <h2 id="history-title" class="panel-title">Recent history · previous 8 records</h2>
            ${historyHtml(observation.recent_history)}
          </section>
        </div>
        <aside class="decision-column" aria-labelledby="decision-title">
          <p class="eyebrow">Current decision</p>
          <h2 id="decision-title">Place order</h2>
          <form id="order-form" novalidate>
            <label class="field">
              <span class="field-label">Quantity · integer 0–128</span>
              <input id="order-input" name="quantity" type="text" inputmode="numeric" autocomplete="off" pattern="[0-9]*" maxlength="3" aria-describedby="order-error">
            </label>
            <p id="order-error" class="error-message" role="alert"></p>
            <div class="decision-actions">
              <button class="primary-button" type="submit">Submit order</button>
              <button id="abandon-button" class="secondary-button" type="button">End session</button>
            </div>
          </form>
          <div class="decision-meta">
            <dl class="state-detail-list">
              <div><dt>Weeks remaining</dt><dd>${observation.weeks_remaining}</dd></div>
              <div><dt>Cost through prior week</dt><dd>${formatCost(costs.cumulative_local_cost_through_previous_week)}</dd></div>
              <div><dt>Holding / backlog</dt><dd>0.5 / 1.0</dd></div>
              <div><dt>Factory capacity</dt><dd>22</dd></div>
            </dl>
          </div>
        </aside>
      </div>
    </section>
  `;
}

function recordFor(status) {
  const completed = status === "completed";
  const grade = completed ? active.episode.outcome.grade : null;
  return {
    session_uuid: active.sessionUuid,
    timestamp: active.timestamp,
    env_version: "0.2.0",
    tier: TIER,
    role: ROLE,
    split: active.seed.split,
    seed_index: active.seed.seed_index,
    seed: active.episode.spec.master_seed_hex,
    scenario_id: active.episode.spec.scenario_id,
    variant: VARIANT,
    prior_beer_game_experience: active.experience,
    status,
    completed,
    actions: [...active.actions],
    weekly: [...active.weekly],
    final_total_cost: completed ? grade.primary.local_total_cost : null,
    base_stock_cost: completed ? grade.primary.paired_base_stock_local_total_cost : null,
    episode_reward: completed ? grade.episode_reward : null,
  };
}

function sendCurrentRecord(status) {
  if (!active || logSent) return false;
  const sent = sendTelemetry(recordFor(status));
  logSent = true;
  return sent;
}

function debriefHtml() {
  const humanGrade = active.episode.outcome.grade;
  const trace = active.trace;
  const llmReplay = replayActions(
    scenarioFor(TIER, active.seed.split, active.seed.seed_index),
    ROLE,
    trace.actions,
  ).episode;
  const llmCost = llmReplay.outcome.grade.primary.local_total_cost;
  if (llmCost !== trace.local_total_cost) {
    throw new Error("recorded LLM trace integrity failure");
  }
  const humanCost = humanGrade.primary.local_total_cost;
  const baseCost = humanGrade.primary.paired_base_stock_local_total_cost;
  return `
    <section class="debrief" aria-labelledby="debrief-title">
      <div class="debrief-head">
        <p class="eyebrow">36 decisions complete · Tier 5 wholesaler</p>
        <h1 id="debrief-title">Your final comparison</h1>
        <p class="lede">Totals include the 36 operational weeks, the three-week deterministic settlement, and terminal inventory-position exposure.</p>
      </div>
      <div class="debrief-body">
        <h2>Same condition. Same seed.</h2>
        <div class="score-grid">
          <article class="score-card you">
            <span class="score-label">You</span>
            <strong id="human-final-cost" class="score-value">${formatCost(humanCost)}</strong>
            <span class="score-note">Local total cost</span>
          </article>
          <article class="score-card">
            <span class="score-label">Recorded LLM</span>
            <strong id="llm-final-cost" class="score-value">${formatCost(llmCost)}</strong>
            <span class="score-note">Recorded evaluation trace</span>
          </article>
          <article class="score-card">
            <span class="score-label">Adaptive base-stock</span>
            <strong id="base-final-cost" class="score-value">${formatCost(baseCost)}</strong>
            <span class="score-note">Paired reference policy</span>
          </article>
        </div>
        <div class="reward-panel">
          <div>
            <p class="score-label">Evaluation reward</p>
            <p class="score-note">Base-stock cost ÷ (base-stock cost + your cost)</p>
          </div>
          <strong id="human-reward" class="reward-value">${humanGrade.episode_reward.toFixed(6)}</strong>
        </div>
        <div class="debrief-actions">
          <button id="new-game-button" class="primary-button" type="button">Play another scenario</button>
        </div>
      </div>
    </section>
  `;
}

function bindBriefing() {
  document.querySelector("#start-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const experience = form.get("experience");
    if (!["yes", "no", "unsure"].includes(experience)) return;
    const random = new Uint32Array(1);
    globalThis.crypto?.getRandomValues?.(random);
    const catalogIndex = random[0] % catalog.seeds.length;
    const seed = catalog.seeds[catalogIndex];
    const spec = scenarioFor(TIER, seed.split, seed.seed_index);
    const episode = new BeerEpisode(spec, ROLE);
    active = {
      seed, catalogIndex, experience, episode,
      observation: episode.start(),
      actions: [],
      weekly: [],
      sessionUuid: createSessionUuid(),
      timestamp: new Date().toISOString(),
      trace: seed,
    };
    logSent = false;
    renderGame();
  });
}

function renderBriefing() {
  active = null;
  logSent = false;
  document.querySelector("#app").innerHTML = briefingHtml();
  setHeader("Briefing");
  bindBriefing();
}

function renderGame() {
  document.querySelector("#app").innerHTML = gameHtml(active.observation);
  setHeader(`Week ${active.observation.week} of 36`);
  const input = document.querySelector("#order-input");
  input.focus();
  input.select();
  document.querySelector("#order-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const parsed = parseOrderInput(input.value);
    const error = document.querySelector("#order-error");
    if (!parsed.ok) {
      error.textContent = parsed.message;
      input.setAttribute("aria-invalid", "true");
      return;
    }
    error.textContent = "";
    input.removeAttribute("aria-invalid");
    const result = active.episode.placeOrder(parsed.quantity);
    active.actions.push(parsed.quantity);
    const transition = active.episode.operationalTransitions.at(-1);
    const state = transition.states_after_fulfillment[ROLE];
    active.weekly.push({
      week: transition.week,
      inventory: state.inventory,
      backlog: state.backlog,
      local_cost: transition.local_costs[ROLE],
    });
    if (result.done) {
      sendCurrentRecord("completed");
      document.querySelector("#app").innerHTML = debriefHtml();
      setHeader("Complete");
      document.querySelector("#new-game-button").addEventListener("click", renderBriefing);
    } else {
      active.observation = result.next_observation;
      renderGame();
    }
  });
  document.querySelector("#abandon-button").addEventListener("click", () => {
    sendCurrentRecord("abandoned");
    renderBriefing();
  });
}

async function loadCatalog() {
  const response = await fetch("./data/llm-comparison.json", { cache: "no-cache" });
  if (!response.ok) throw new Error(`trace catalog returned ${response.status}`);
  const payload = await response.json();
  if (!Array.isArray(payload.seeds) || payload.seeds.length !== 8) {
    throw new Error("trace catalog must contain exactly eight headline seeds");
  }
  return payload;
}

export async function initialize() {
  configureTelemetry(globalThis.BEER_GAME_CONFIG?.loggingEndpoint || "");
  try {
    catalog = await loadCatalog();
    renderBriefing();
  } catch (error) {
    console.error(error);
    document.querySelector("#app").innerHTML = `
      <section class="loading-card">
        <p class="eyebrow">Unable to start</p>
        <h1>Scenario data could not be loaded.</h1>
        <p class="lede">Reload the page to try again.</p>
      </section>
    `;
    setHeader("Unavailable");
  }
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  window.addEventListener("pagehide", () => {
    if (active && !active.episode.done) {
      sendCurrentRecord("abandoned");
    }
  });
  void initialize();
}
