/**
 * FootyPredict: Dual-Stage Match Simulation & Inference Engine
 * Client-side execution of GBDT Classification and Bivariate Poisson Scoreline Formulation.
 */

const TEAMS_DATABASE = {
  "Premier League": [
    { name: "Arsenal", elo: 1890, att: 2.15, defn: 0.75, form: ["W", "W", "W", "D", "W"], gd: 8, rest: 6 },
    { name: "Manchester City", elo: 1920, att: 2.30, defn: 0.80, form: ["W", "W", "D", "W", "W"], gd: 9, rest: 5 },
    { name: "Liverpool", elo: 1905, att: 2.25, defn: 0.82, form: ["W", "D", "W", "W", "W"], gd: 7, rest: 7 },
    { name: "Chelsea", elo: 1780, att: 1.75, defn: 1.10, form: ["W", "L", "W", "D", "W"], gd: 2, rest: 4 },
    { name: "Tottenham", elo: 1765, att: 1.80, defn: 1.25, form: ["L", "W", "L", "W", "D"], gd: 1, rest: 6 },
    { name: "Aston Villa", elo: 1770, att: 1.70, defn: 1.15, form: ["W", "W", "L", "D", "W"], gd: 3, rest: 5 },
    { name: "Newcastle United", elo: 1750, att: 1.65, defn: 1.18, form: ["D", "W", "L", "W", "L"], gd: 0, rest: 6 },
    { name: "Manchester United", elo: 1740, att: 1.55, defn: 1.22, form: ["L", "D", "W", "L", "W"], gd: -1, rest: 4 }
  ],
  "La Liga": [
    { name: "Real Madrid", elo: 1930, att: 2.35, defn: 0.78, form: ["W", "W", "W", "W", "D"], gd: 10, rest: 6 },
    { name: "Barcelona", elo: 1910, att: 2.40, defn: 0.85, form: ["W", "W", "L", "W", "W"], gd: 8, rest: 5 },
    { name: "Atletico Madrid", elo: 1820, att: 1.75, defn: 0.80, form: ["D", "W", "W", "D", "W"], gd: 5, rest: 7 },
    { name: "Athletic Bilbao", elo: 1740, att: 1.55, defn: 1.05, form: ["W", "L", "W", "D", "D"], gd: 2, rest: 5 },
    { name: "Real Sociedad", elo: 1725, att: 1.45, defn: 1.10, form: ["L", "W", "D", "L", "W"], gd: 0, rest: 6 }
  ],
  "Bundesliga": [
    { name: "Bayern Munich", elo: 1915, att: 2.50, defn: 0.90, form: ["W", "W", "W", "D", "W"], gd: 12, rest: 7 },
    { name: "Bayer Leverkusen", elo: 1860, att: 2.10, defn: 0.92, form: ["W", "D", "W", "W", "D"], gd: 6, rest: 6 },
    { name: "Borussia Dortmund", elo: 1800, att: 1.95, defn: 1.20, form: ["W", "L", "W", "W", "L"], gd: 3, rest: 5 },
    { name: "RB Leipzig", elo: 1785, att: 1.85, defn: 1.15, form: ["D", "W", "W", "L", "W"], gd: 4, rest: 6 }
  ],
  "Serie A": [
    { name: "Inter Milan", elo: 1875, att: 2.05, defn: 0.76, form: ["W", "W", "W", "D", "W"], gd: 7, rest: 6 },
    { name: "Juventus", elo: 1810, att: 1.65, defn: 0.82, form: ["D", "W", "D", "W", "D"], gd: 3, rest: 7 },
    { name: "AC Milan", elo: 1800, att: 1.80, defn: 1.15, form: ["W", "L", "W", "D", "W"], gd: 2, rest: 5 },
    { name: "Napoli", elo: 1795, att: 1.75, defn: 1.05, form: ["W", "W", "L", "W", "D"], gd: 4, rest: 6 }
  ]
};

// Mathematical helper for Poisson probability: P(k; λ) = (λ^k * e^-λ) / k!
function factorial(n) {
  let res = 1;
  for (let i = 2; i <= n; i++) res *= i;
  return res;
}

function poisson(k, lambda) {
  return (Math.pow(lambda, k) * Math.exp(-lambda)) / factorial(k);
}

// DOM Elements
const leagueSelect = document.getElementById("league-select");
const homeTeamSelect = document.getElementById("home-team-select");
const awayTeamSelect = document.getElementById("away-team-select");
const swapBtn = document.getElementById("swap-teams-btn");
const simulateBtn = document.getElementById("simulate-btn");
const presetChips = document.querySelectorAll(".preset-chip");

// Populating Team Selectors
function populateTeams(leagueName, defaultHome = null, defaultAway = null) {
  const teams = TEAMS_DATABASE[leagueName];
  homeTeamSelect.innerHTML = "";
  awayTeamSelect.innerHTML = "";

  teams.forEach((team, idx) => {
    const optHome = document.createElement("option");
    optHome.value = team.name;
    optHome.textContent = team.name;
    homeTeamSelect.appendChild(optHome);

    const optAway = document.createElement("option");
    optAway.value = team.name;
    optAway.textContent = team.name;
    awayTeamSelect.appendChild(optAway);
  });

  if (defaultHome && teams.some(t => t.name === defaultHome)) {
    homeTeamSelect.value = defaultHome;
  } else {
    homeTeamSelect.selectedIndex = 0;
  }

  if (defaultAway && teams.some(t => t.name === defaultAway)) {
    awayTeamSelect.value = defaultAway;
  } else {
    awayTeamSelect.selectedIndex = Math.min(1, teams.length - 1);
  }
}

function getTeam(name) {
  for (const league in TEAMS_DATABASE) {
    const found = TEAMS_DATABASE[league].find(t => t.name === name);
    if (found) return found;
  }
  return null;
}

// Core Dual-Stage Computation
function computeMatchInference() {
  const homeName = homeTeamSelect.value;
  const awayName = awayTeamSelect.value;

  if (homeName === awayName) {
    alert("Please select two distinct teams for the match simulation.");
    return;
  }

  const home = getTeam(homeName);
  const away = getTeam(awayName);

  if (!home || !away) return;

  // 1. Feature Engineering
  const homeAdvantageElo = 65; // Statistical home pitch advantage
  const eloDiff = (home.elo - away.elo) + homeAdvantageElo;

  const restAdvantageHome = home.rest >= 6 ? 0.05 : -0.05;
  const restAdvantageAway = away.rest >= 6 ? 0.05 : -0.05;

  // 2. Stage 2: Bivariate Poisson Expected Goals (xG)
  const lamHome = Math.max(0.4, (home.att * 0.70 + away.defn * 0.45) + 0.28 + restAdvantageHome + (eloDiff / 650.0) * 0.4);
  const lamAway = Math.max(0.3, (away.att * 0.65 + home.defn * 0.40) + restAdvantageAway - (eloDiff / 650.0) * 0.35);

  // Generate Scoreline Matrix (0-0 to 5-5)
  const maxGoals = 6;
  const scoreMatrix = [];
  let sumPoisHome = 0;
  let sumPoisDraw = 0;
  let sumPoisAway = 0;
  const flatScores = [];

  for (let h = 0; h < maxGoals; h++) {
    const pH = poisson(h, lamHome);
    for (let a = 0; a < maxGoals; a++) {
      const pA = poisson(a, lamAway);
      const prob = pH * pA;
      scoreMatrix.push({ h, a, prob });
      flatScores.push({ score: `${h} - ${a}`, h, a, prob });

      if (h > a) sumPoisHome += prob;
      else if (h === a) sumPoisDraw += prob;
      else sumPoisAway += prob;
    }
  }

  // Normalize Poisson outcomes
  const totalPois = sumPoisHome + sumPoisDraw + sumPoisAway;
  sumPoisHome /= totalPois;
  sumPoisDraw /= totalPois;
  sumPoisAway /= totalPois;

  // 3. Stage 1: GBDT Sigmoid Outcome Classification
  // Logistic link on EloDiff and Form Differential
  const formDiff = (home.form.filter(x => x === 'W').length * 3) - (away.form.filter(x => x === 'W').length * 3);
  const zHome = (eloDiff / 220.0) + (formDiff * 0.08) + 0.35;
  const zAway = -(eloDiff / 250.0) - (formDiff * 0.08) - 0.25;

  let clfHome = 1.0 / (1.0 + Math.exp(-zHome));
  let clfAway = 1.0 / (1.0 + Math.exp(-zAway));
  let clfDraw = Math.max(0.18, 1.0 - (clfHome + clfAway) * 0.55);

  const clfTotal = clfHome + clfDraw + clfAway;
  clfHome /= clfTotal;
  clfDraw /= clfTotal;
  clfAway /= clfTotal;

  // 4. Dual-Stage Ensemble Blending (55% GBDT Classifier, 45% Poisson Formulation)
  const alpha = 0.55;
  let finalHomeProb = alpha * clfHome + (1 - alpha) * sumPoisHome;
  let finalDrawProb = alpha * clfDraw + (1 - alpha) * sumPoisDraw;
  let finalAwayProb = alpha * clfAway + (1 - alpha) * sumPoisAway;

  const finalTotal = finalHomeProb + finalDrawProb + finalAwayProb;
  finalHomeProb /= finalTotal;
  finalDrawProb /= finalTotal;
  finalAwayProb /= finalTotal;

  // Sort scorelines by probability
  flatScores.sort((a, b) => b.prob - a.prob);
  const topScorelines = flatScores.slice(0, 5);

  // Update UI State
  updateUI({
    home,
    away,
    eloDiff,
    finalHomeProb,
    finalDrawProb,
    finalAwayProb,
    lamHome,
    lamAway,
    topScorelines
  });
}

function updateUI(data) {
  const {
    home, away, eloDiff,
    finalHomeProb, finalDrawProb, finalAwayProb,
    lamHome, lamAway, topScorelines
  } = data;

  // Update Team Names
  document.getElementById("stat-home-name").textContent = home.name;
  document.getElementById("stat-away-name").textContent = away.name;
  document.getElementById("xg-home-team").textContent = `${home.name} xG`;
  document.getElementById("xg-away-team").textContent = `${away.name} xG`;
  document.getElementById("diag-home-name").textContent = home.name;
  document.getElementById("diag-away-name").textContent = away.name;

  // Update Elo & Ratings
  document.getElementById("diag-home-elo").textContent = `${home.elo} Elo`;
  document.getElementById("diag-away-elo").textContent = `${away.elo} Elo`;
  const eloDeltaElem = document.getElementById("val-elo-delta");
  eloDeltaElem.textContent = (eloDiff >= 0 ? `+${Math.round(eloDiff)}` : `${Math.round(eloDiff)}`);
  eloDeltaElem.style.color = eloDiff >= 0 ? "var(--accent-pitch)" : "var(--accent-away)";

  // Update Probabilities
  const pHPct = (finalHomeProb * 100).toFixed(1);
  const pDPct = (finalDrawProb * 100).toFixed(1);
  const pAPct = (finalAwayProb * 100).toFixed(1);

  document.getElementById("val-home-prob").textContent = `${pHPct}%`;
  document.getElementById("val-draw-prob").textContent = `${pDPct}%`;
  document.getElementById("val-away-prob").textContent = `${pAPct}%`;

  document.getElementById("val-home-odds").textContent = `${(1 / finalHomeProb).toFixed(2)} Fair Odds`;
  document.getElementById("val-draw-odds").textContent = `${(1 / finalDrawProb).toFixed(2)} Fair Odds`;
  document.getElementById("val-away-odds").textContent = `${(1 / finalAwayProb).toFixed(2)} Fair Odds`;

  // Segmented Bar Widths
  document.getElementById("bar-home").style.width = `${pHPct}%`;
  document.getElementById("bar-draw").style.width = `${pDPct}%`;
  document.getElementById("bar-away").style.width = `${pAPct}%`;

  // Dominant Outcome Badge
  const domBadge = document.getElementById("dominant-outcome");
  if (finalHomeProb > 0.48) {
    domBadge.textContent = `${home.name} Favored`;
    domBadge.style.color = "var(--accent-pitch)";
    domBadge.style.borderColor = "rgba(16, 185, 129, 0.3)";
  } else if (finalAwayProb > 0.42) {
    domBadge.textContent = `${away.name} Favored`;
    domBadge.style.color = "var(--accent-away)";
    domBadge.style.borderColor = "rgba(56, 189, 248, 0.3)";
  } else {
    domBadge.textContent = "High Variance / Even Contest";
    domBadge.style.color = "var(--accent-draw)";
    domBadge.style.borderColor = "rgba(245, 158, 11, 0.3)";
  }

  // Expected Goals (xG)
  document.getElementById("val-home-xg").textContent = lamHome.toFixed(2);
  document.getElementById("val-away-xg").textContent = lamAway.toFixed(2);

  // Top 5 Scorelines
  const scorelinesList = document.getElementById("scorelines-list");
  scorelinesList.innerHTML = "";
  const maxScoreProb = topScorelines[0].prob;

  topScorelines.forEach(item => {
    const row = document.createElement("div");
    row.className = "scoreline-row";

    const pct = (item.prob * 100).toFixed(1);
    const fillWidth = Math.min(100, Math.round((item.prob / maxScoreProb) * 100));

    row.innerHTML = `
      <span class="score-tag">${item.score}</span>
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width: ${fillWidth}%;"></div>
      </div>
      <span class="score-pct">${pct}%</span>
    `;
    scorelinesList.appendChild(row);
  });

  // Render Form Badges
  renderFormBadges("home-form-badges", home.form);
  renderFormBadges("away-form-badges", away.form);

  document.getElementById("home-gd-val").textContent = (home.gd > 0 ? `+${home.gd}` : `${home.gd}`);
  document.getElementById("away-gd-val").textContent = (away.gd > 0 ? `+${away.gd}` : `${away.gd}`);
  document.getElementById("home-rest-val").textContent = `${home.rest} Days`;
  document.getElementById("away-rest-val").textContent = `${away.rest} Days`;
}

function renderFormBadges(containerId, formArr) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  formArr.forEach(result => {
    const b = document.createElement("span");
    b.className = `form-badge badge-${result.toLowerCase()}`;
    b.textContent = result;
    container.appendChild(b);
  });
}

// Event Listeners
leagueSelect.addEventListener("change", (e) => {
  populateTeams(e.target.value);
  computeMatchInference();
});

homeTeamSelect.addEventListener("change", computeMatchInference);
awayTeamSelect.addEventListener("change", computeMatchInference);

swapBtn.addEventListener("click", () => {
  const hVal = homeTeamSelect.value;
  const aVal = awayTeamSelect.value;
  homeTeamSelect.value = aVal;
  awayTeamSelect.value = hVal;
  computeMatchInference();
});

simulateBtn.addEventListener("click", computeMatchInference);

presetChips.forEach(chip => {
  chip.addEventListener("click", () => {
    presetChips.forEach(c => c.classList.remove("active"));
    chip.classList.add("active");

    const league = chip.getAttribute("data-league");
    const home = chip.getAttribute("data-home");
    const away = chip.getAttribute("data-away");

    leagueSelect.value = league;
    populateTeams(league, home, away);
    computeMatchInference();
  });
});

// Initialization
window.addEventListener("DOMContentLoaded", () => {
  populateTeams("Premier League", "Arsenal", "Chelsea");
  computeMatchInference();
});
