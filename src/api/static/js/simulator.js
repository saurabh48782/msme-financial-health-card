import { themed, makeChart, PALETTE, rupees } from './charts.js';

const form = document.getElementById('simulate-form');
if (form) {
  const msmeId = form.dataset.msmeId;
  const resultCard = document.getElementById('result-card');
  const resultError = document.getElementById('result-error');
  const rows = document.getElementById('result-rows');
  const grades = document.getElementById('result-grades');
  const alertBox = document.getElementById('result-alert');
  let chart = null;

  // Live value readout beside each slider.
  form.querySelectorAll('input[type=range]').forEach((input) => {
    const output = document.getElementById(`out-${input.name}`);
    const sync = () => {
      if (output) output.textContent = Number(input.value).toFixed(2);
      const changed = Number(input.value) !== Number(input.dataset.original);
      input.classList.toggle('border-primary', changed);
    };
    input.addEventListener('input', sync);
    sync();
  });

  document.getElementById('reset-levers')?.addEventListener('click', () => {
    form.querySelectorAll('input[type=range]').forEach((input) => {
      input.value = input.dataset.original;
      input.dispatchEvent(new Event('input'));
    });
    resultCard.hidden = true;
    resultError.hidden = true;
  });

  const FORMATTERS = {
    financial_health_score: (v) => v.toFixed(2),
    probability_of_default: (v) => `${(v * 100).toFixed(2)}%`,
    credit_limit_inr: rupees,
    tenor_months: (v) => `${v} months`,
    indicative_rate_pct: (v) => `${v.toFixed(2)}%`,
  };
  // The row label defaults to the metric key; only names that read badly are overridden.
  const LABELS = { tenor_months: 'tenure (months)' };
  // Lower is better for these, so a negative change is good news.
  const LOWER_IS_BETTER = new Set(['probability_of_default', 'indicative_rate_pct']);

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    resultError.hidden = true;

    // Send only what actually moved.
    const overrides = {};
    form.querySelectorAll('input[type=range]').forEach((input) => {
      if (Number(input.value) !== Number(input.dataset.original)) {
        overrides[input.name] = Number(input.value);
      }
    });
    if (!Object.keys(overrides).length) {
      resultError.textContent = 'Move at least one lever before re-scoring.';
      resultError.hidden = false;
      return;
    }

    const button = form.querySelector('button[type=submit]');
    button.disabled = true;
    button.textContent = 'Scoring…';
    try {
      const response = await fetch(`/api/v1/simulate/${encodeURIComponent(msmeId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ overrides }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed (${response.status})`);
      }
      render(await response.json());
    } catch (error) {
      resultError.textContent = error.message;
      resultError.hidden = false;
      resultCard.hidden = true;
    } finally {
      button.disabled = false;
      button.textContent = 'Re-score';
    }
  });

  function render(result) {
    rows.innerHTML = '';
    for (const delta of result.deltas) {
      const format = FORMATTERS[delta.metric] || ((v) => v.toFixed(2));
      const improved = LOWER_IS_BETTER.has(delta.metric) ? delta.delta < 0 : delta.delta > 0;
      const tone = delta.delta === 0 ? 'secondary' : improved ? 'success' : 'danger';
      const sign = delta.delta > 0 ? '+' : '';
      rows.insertAdjacentHTML('beforeend', `
        <tr>
          <td>${LABELS[delta.metric] || delta.metric.replace(/_/g, ' ')}</td>
          <td class="text-end font-monospace">${format(delta.before)}</td>
          <td class="text-end font-monospace fw-semibold">${format(delta.after)}</td>
          <td class="text-end text-${tone} font-monospace">${sign}${format(delta.delta)}</td>
        </tr>`);
    }

    grades.textContent = `${result.before_grade} → ${result.after_grade}`;
    const bandBefore = result.before_credit.risk_band;
    const bandAfter = result.after_credit.risk_band;
    if (result.unchanged) {
      alertBox.innerHTML = '<div class="alert alert-secondary mb-3">These changes did not move the outcome.</div>';
    } else if (!result.before_credit.eligible && result.after_credit.eligible) {
      alertBox.innerHTML = `<div class="alert alert-success mb-3">
        <strong>This firm would become eligible.</strong> Risk band moves ${bandBefore} → ${bandAfter},
        unlocking a limit of ${rupees(result.after_credit.credit_limit_inr)}.</div>`;
    } else if (result.before_credit.eligible && !result.after_credit.eligible) {
      alertBox.innerHTML = `<div class="alert alert-danger mb-3">
        <strong>This firm would lose eligibility.</strong> Risk band moves ${bandBefore} → ${bandAfter}.</div>`;
    } else {
      alertBox.innerHTML = `<div class="alert alert-info mb-3">Risk band ${bandBefore} → ${bandAfter}.</div>`;
    }

    const labels = result.after_pillars.map((p) => p.label);
    const config = {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Before', data: result.before_pillars.map((p) => p.score), backgroundColor: PALETTE.secondary, borderRadius: 3 },
          { label: 'After', data: result.after_pillars.map((p) => p.score), backgroundColor: PALETTE.primary, borderRadius: 3 },
        ],
      },
      options: themed({ scales: { y: { beginAtZero: true, max: 100 } } }),
    };
    if (chart) chart.destroy();
    resultCard.hidden = false;
    chart = makeChart('simPillars', config);
  }
}
