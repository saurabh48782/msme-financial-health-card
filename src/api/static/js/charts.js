// Shared Chart.js helpers. Vanilla ES modules, no bundler.
//
// One rule enforced here: every chart reads its data from a <script
// type="application/json"> block rendered by the server, never from a fetch on
// page load. The page is therefore complete and correct before any JavaScript
// runs, and the charts are an enhancement rather than a dependency.

export const PALETTE = {
  primary: '#0d6efd', success: '#198754', warning: '#ffc107',
  danger: '#dc3545', info: '#0dcaf0', secondary: '#6c757d',
  purple: '#6f42c1', teal: '#20c997', orange: '#fd7e14', pink: '#d63384',
};

export const SERIES = [
  PALETTE.primary, PALETTE.teal, PALETTE.orange, PALETTE.purple,
  PALETTE.pink, PALETTE.info, PALETTE.success, PALETTE.warning,
  PALETTE.danger, PALETTE.secondary,
];

export const BAND_COLOURS = {
  Low: PALETTE.success, Medium: PALETTE.warning, High: PALETTE.danger,
};

export const GRADE_COLOURS = {
  'A+': PALETTE.success, A: PALETTE.teal, B: PALETTE.primary,
  C: PALETTE.warning, D: PALETTE.danger,
};

/** Read a server-rendered JSON payload by element id. */
export function payload(id) {
  const node = document.getElementById(id);
  if (!node) return null;
  try {
    return JSON.parse(node.textContent);
  } catch (error) {
    console.error(`Could not parse payload #${id}`, error);
    return null;
  }
}

/** Chart defaults that follow the Bootstrap theme rather than fighting it. */
export function themed(options = {}) {
  const styles = getComputedStyle(document.body);
  const text = styles.getPropertyValue('--bs-body-color').trim() || '#212529';
  const grid = styles.getPropertyValue('--bs-border-color').trim() || '#dee2e6';
  return {
    responsive: true,
    maintainAspectRatio: false,
    color: text,
    plugins: {
      legend: { labels: { color: text, boxWidth: 12, boxHeight: 12 } },
      tooltip: { intersect: false, mode: 'index' },
      ...(options.plugins || {}),
    },
    scales: options.scales
      ? Object.fromEntries(
          Object.entries(options.scales).map(([key, scale]) => [
            key,
            { ticks: { color: text }, grid: { color: grid }, ...scale },
          ]),
        )
      : undefined,
    ...Object.fromEntries(Object.entries(options).filter(([k]) => k !== 'plugins' && k !== 'scales')),
  };
}

export function makeChart(canvasId, config) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  return new Chart(canvas, config);
}

export const percent = (v) => `${(v * 100).toFixed(1)}%`;

export function rupees(value) {
  const amount = Number(value) || 0;
  if (Math.abs(amount) >= 1e7) return `₹${(amount / 1e7).toFixed(2)} cr`;
  if (Math.abs(amount) >= 1e5) return `₹${(amount / 1e5).toFixed(2)} L`;
  return `₹${amount.toLocaleString('en-IN')}`;
}
