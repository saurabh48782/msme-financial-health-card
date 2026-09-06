// The assessment desk. Collects the feature form, posts it to /api/v1/score and
// renders the card the API returns.
//
// Unlike the other pages, this one has no server-rendered payload to read: the
// firm does not exist until the analyst types it, so the verdict can only come
// from the API. That is the point — the page shows exactly the card an
// integrator receives, not a second rendering of the same decision.
import { themed, makeChart, PALETTE, rupees } from './charts.js';

const form = document.getElementById('assess-form');
if (form) {
  const placeholder = document.getElementById('assess-placeholder');
  const result = document.getElementById('assess-result');
  const errorBox = document.getElementById('assess-error');
  let chart = null;

  const GRADE_CLASS = { 'A+': 'success', A: 'success', B: 'primary', C: 'warning', D: 'danger' };
  const BAND_CLASS = { Low: 'success', Medium: 'warning', High: 'danger' };
  const SEVERITY_CLASS = { critical: 'danger', high: 'danger', medium: 'warning', low: 'info' };
  const MONEYISH = /inr|cashflow|emi|principal/;

  const esc = (value) =>
    String(value ?? '').replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const pct = (v, places = 2) => `${(Number(v) * 100).toFixed(places)}%`;
  const num = (v, places = 2) => (v === null || v === undefined ? '—' : Number(v).toFixed(places));

  document.getElementById('reset-form').addEventListener('click', () => {
    form.reset();
    result.hidden = true;
    errorBox.hidden = true;
    placeholder.hidden = false;
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    errorBox.hidden = true;
    if (!form.reportValidity()) return;

    const button = form.querySelector('button[type=submit]');
    button.disabled = true;
    button.textContent = 'Assessing…';
    try {
      const response = await fetch('/api/v1/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ records: [collect()], include_explanation: true }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(detailOf(body, response.status));
      render(body[0]);
    } catch (error) {
      errorBox.innerHTML = error.message;
      errorBox.hidden = false;
      result.hidden = true;
      placeholder.hidden = false;
    } finally {
      button.disabled = false;
      button.textContent = 'Assess this firm';
    }
  });

  /** Read the form into one MSMEFeatures record, keyed by dataset column name. */
  function collect() {
    const record = {};
    for (const input of form.querySelectorAll('[name]')) {
      const value = input.value.trim();
      // An empty optional input means "not reported", which is a null the rubric
      // redistributes — not a zero, which would be a repayment record of zero.
      if (value === '') {
        if (!input.required) record[input.name] = null;
        continue;
      }
      record[input.name] = input.dataset.kind === 'number' ? Number(value) : value;
    }
    return record;
  }

  /** FastAPI's 422 names the offending field; a flat string would waste that. */
  function detailOf(body, status) {
    const detail = body.detail;
    if (Array.isArray(detail)) {
      const lines = detail.map((item) => {
        const field = (item.loc || []).slice(-1)[0];
        return `<li><code>${esc(field)}</code> — ${esc(item.msg)}</li>`;
      });
      return `<strong>The payload was rejected.</strong><ul class="mb-0 mt-2">${lines.join('')}</ul>`;
    }
    return esc(detail || `Request failed (${status})`);
  }

  function render(card) {
    const credit = card.credit;
    document.getElementById('result-id').textContent =
      `${card.msme_id} · scored ${new Date(card.scored_at).toLocaleString()}`;

    renderVerdict(card, credit);
    renderOffer(card, credit);
    renderReasons(credit);
    renderBasis(credit);
    renderPillars(card.pillars, card.financial_health_score);
    renderReasonCodes(card.explanation);
    renderFlags(card.risk_flags);
    renderProvenance(card);

    placeholder.hidden = true;
    result.hidden = false;
    result.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderVerdict(card, credit) {
    const box = document.getElementById('result-verdict');
    const thin = card.thin_file
      ? ' Scored with no formal credit history — the repayment weight was redistributed across the evidence that does exist.'
      : '';
    box.innerHTML = credit.eligible
      ? `<div class="alert alert-success">
           <strong>A loan can be provided.</strong> This firm clears the
           ${esc(credit.policy_version)} policy at a ${esc(credit.risk_band)} risk band.${thin}
         </div>`
      : `<div class="alert alert-danger">
           <strong>A loan cannot be provided under the current policy.</strong>
           The estimated default probability of ${pct(credit.probability_of_default)} places this
           firm in the ${esc(credit.risk_band)} risk band.${thin}
         </div>`;
  }

  function renderOffer(card, credit) {
    const tile = (label, value, sub = '', tone = 'body') => `
      <div class="col">
        <div class="text-body-secondary small text-uppercase fw-semibold">${label}</div>
        <div class="fs-4 fw-semibold text-${tone}">${value}</div>
        <div class="small">${sub}</div>
      </div>`;
    const gradeClass = GRADE_CLASS[card.grade] || 'secondary';
    const bandClass = BAND_CLASS[credit.risk_band] || 'secondary';
    document.getElementById('result-offer').innerHTML = [
      tile('Health score', num(card.financial_health_score),
        `<span class="badge text-bg-${gradeClass}">${esc(card.grade)} · ${esc(card.grade_label)}</span>`,
        gradeClass),
      tile('Default probability', pct(credit.probability_of_default),
        `<span class="badge text-bg-${bandClass}">${esc(credit.risk_band)} risk</span>`),
      tile('Sanctioned limit', credit.eligible ? rupees(credit.credit_limit_inr) : '—',
        credit.eligible ? '' : 'nothing sanctioned',
        credit.eligible ? 'success' : 'body-secondary'),
      tile('Terms', credit.eligible ? `${credit.tenor_months} mo` : '—',
        credit.eligible ? `at ${num(credit.indicative_rate_pct)}% indicative` : ''),
    ].join('');
  }

  function renderReasons(credit) {
    const box = document.getElementById('result-reasons');
    const parts = [];
    if (credit.decline_reasons.length) {
      parts.push(`<div class="alert alert-secondary mb-2">
        <div class="fw-semibold">Why the policy declined</div>
        <ul class="mb-0 mt-2 small">
          ${credit.decline_reasons.map((r) => `<li>${esc(r)}</li>`).join('')}
        </ul></div>`);
    }
    // A policy decline the classifier disagrees with is a manual-review case, not
    // a silent rejection — so it is said out loud rather than left in the JSON.
    const p = credit.model_eligibility_probability;
    if (p !== null && p !== undefined && !credit.eligible && p >= 0.5) {
      parts.push(`<div class="alert alert-warning mb-0 small">
        The eligibility model puts this firm at ${pct(p, 1)} likely to qualify while the policy
        declines it. A disagreement of that size belongs in manual review.</div>`);
    }
    box.innerHTML = parts.join('');
  }

  function renderBasis(credit) {
    const wrap = document.getElementById('result-basis-wrap');
    const entries = Object.entries(credit.limit_basis || {});
    wrap.hidden = entries.length === 0;
    document.getElementById('result-basis').innerHTML = entries
      .map(([key, value]) => `
        <tr>
          <td class="text-body-secondary">${esc(key.replace(/_/g, ' '))}</td>
          <td class="text-end font-monospace">${MONEYISH.test(key) ? rupees(value) : num(value, 4)}</td>
        </tr>`)
      .join('');
    document.getElementById('result-notes').innerHTML = (credit.notes || [])
      .map((note) => `<div>${esc(note)}</div>`)
      .join('');
  }

  function renderPillars(pillars, score) {
    document.getElementById('result-pillars').innerHTML = pillars
      .map((pillar, index) => {
        const thin = pillar.thin_file
          ? ' <span class="badge text-bg-info" title="Repayment driver unavailable; its weight was redistributed">thin file</span>'
          : '';
        const drivers = pillar.drivers
          .map((driver) => `
            <tr class="driver-row small ${driver.available ? '' : 'driver-unavailable'}">
              <td>${esc(driver.label)}${driver.available ? '' : ' <span class="badge text-bg-light border">not reported</span>'}</td>
              <td class="text-end font-monospace">${esc(driver.formatted_value || '—')}</td>
              <td class="text-end">${num(driver.driver_score, 1)}</td>
              <td class="text-end text-body-secondary">${pct(driver.weight, 1)}</td>
              <td class="text-end">${num(driver.contribution)}</td>
            </tr>`)
          .join('');
        return `
          <tr>
            <td>
              <a class="text-decoration-none" data-bs-toggle="collapse" role="button"
                 href="#assess-drivers-${index}" aria-expanded="false"
                 aria-controls="assess-drivers-${index}">${esc(pillar.label)}</a>${thin}
            </td>
            <td class="text-end fw-semibold">${num(pillar.score)}</td>
            <td class="text-end text-body-secondary">${pct(pillar.weight, 1)}</td>
            <td class="text-end">${num(pillar.contribution)}</td>
          </tr>
          <tr class="collapse" id="assess-drivers-${index}">
            <td colspan="4" class="bg-body-tertiary">
              <div class="table-scroll">
                <table class="table table-sm mb-0">
                  <thead><tr class="small text-body-secondary">
                    <th>Driver</th><th class="text-end">Observed</th><th class="text-end">Driver score</th>
                    <th class="text-end">Weight</th><th class="text-end">Points</th>
                  </tr></thead>
                  <tbody>
                    <tr class="driver-row small">
                      <td class="text-body-secondary fst-italic">peer-market baseline</td>
                      <td class="text-end">—</td><td class="text-end">—</td><td class="text-end">—</td>
                      <td class="text-end">${num(pillar.baseline)}</td>
                    </tr>
                    ${drivers}
                  </tbody>
                </table>
              </div>
            </td>
          </tr>`;
      })
      .join('') + `
        <tr class="fw-semibold border-top">
          <td>Financial Health Score</td><td class="text-end">${num(score)}</td>
          <td class="text-end text-body-secondary">100.0%</td><td class="text-end">${num(score)}</td>
        </tr>`;

    if (chart) chart.destroy();
    chart = makeChart('assessPillars', {
      type: 'bar',
      data: {
        labels: pillars.map((p) => p.label),
        datasets: [{
          label: 'Pillar score',
          data: pillars.map((p) => p.score),
          backgroundColor: pillars.map((p) =>
            p.score >= 70 ? PALETTE.success : p.score >= 50 ? PALETTE.warning : PALETTE.danger),
          borderRadius: 3,
        }],
      },
      options: themed({
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, max: 100 } },
      }),
    });
  }

  function renderReasonCodes(explanation) {
    const codes = explanation?.reason_codes || [];
    const fill = (id, wanted, empty) => {
      const items = codes.filter((code) => code.direction === wanted);
      document.getElementById(id).innerHTML = items.length
        ? items
            .map((code) => `
              <li class="list-group-item reason-${wanted}">
                ${esc(code.text)}
                <span class="badge text-bg-light border float-end">${esc(code.source)}</span>
              </li>`)
            .join('')
        : `<li class="list-group-item text-body-secondary">${empty}</li>`;
    };
    fill('result-strengths', 'positive', 'No material strengths identified.');
    fill('result-weaknesses', 'negative', 'No material weaknesses identified.');
  }

  function renderFlags(flags) {
    document.getElementById('result-flags').innerHTML = flags.length
      ? flags
          .map((flag) => `
            <li class="list-group-item d-flex justify-content-between align-items-start gap-2">
              <div>
                <div>${esc(flag.message)}</div>
                <div class="small text-body-secondary font-monospace">
                  ${esc(flag.metric)} = ${num(flag.value, 3)} (threshold ${num(flag.threshold)})
                </div>
              </div>
              <span class="badge text-bg-${SEVERITY_CLASS[flag.severity] || 'secondary'}">${esc(flag.severity)}</span>
            </li>`)
          .join('')
      : '<li class="list-group-item text-body-secondary">No consistency rules triggered. GST, bank and UPI evidence agree.</li>';
  }

  function renderProvenance(card) {
    const item = (label, value) => `
      <div class="col">
        <div class="text-body-secondary text-uppercase fw-semibold">${label}</div>
        <code>${esc(value)}</code>
      </div>`;
    document.getElementById('result-provenance').innerHTML = [
      item('Model version', card.model_version || 'rubric-only (no model loaded)'),
      item('Policy version', card.policy_version),
      item('Rubric version', card.rubric_version),
      item('Feature snapshot', card.feature_snapshot_hash),
    ].join('');
  }
}
