import { payload, themed, makeChart, PALETTE } from './charts.js';

const data = payload('metrics-data');
if (data) {
  // Reliability: the diagonal is perfect calibration, the line is the model.
  if (data.reliability?.length) {
    const points = data.reliability.map((row) => ({ x: row.mean_predicted, y: row.mean_actual }));
    const max = Math.max(...points.flatMap((p) => [p.x, p.y]), 0.01);
    makeChart('reliabilityChart', {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Perfect calibration',
            data: [{ x: 0, y: 0 }, { x: max, y: max }],
            type: 'line',
            borderColor: PALETTE.secondary,
            borderDash: [6, 4],
            pointRadius: 0,
            borderWidth: 1,
          },
          {
            label: 'Model',
            data: points,
            type: 'line',
            borderColor: PALETTE.primary,
            backgroundColor: PALETTE.primary,
            pointRadius: 4,
            tension: 0.2,
          },
        ],
      },
      options: themed({
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) =>
                `predicted ${(ctx.parsed.x * 100).toFixed(2)}% vs realised ${(ctx.parsed.y * 100).toFixed(2)}%`,
            },
          },
        },
        scales: {
          x: { title: { display: true, text: 'Mean predicted PD' }, min: 0, max },
          y: { title: { display: true, text: 'Mean realised PD' }, min: 0, max },
        },
      }),
    });
  }

  // Pillar adjusted R² against the gate, so a marginal pillar is visible at a glance.
  // Reports written before the gate moved to adjusted R² carry no adj_r2; skip
  // them rather than plotting undefined.
  const keys = Object.keys(data.pillars || {}).filter(
    (k) => typeof data.pillars[k].adj_r2 === 'number',
  );
  if (keys.length) {
    const threshold = data.pillar_adj_r2_min ?? 0.75;
    const values = keys.map((k) => data.pillars[k].adj_r2);
    makeChart('pillarFit', {
      type: 'bar',
      data: {
        labels: keys.map((k) => data.labels?.[k] || k),
        datasets: [
          {
            label: 'Adjusted R² vs the dataset pillar',
            data: values,
            backgroundColor: values.map((v) => (v >= threshold ? PALETTE.success : PALETTE.danger)),
            borderRadius: 3,
          },
          {
            label: `Gate (${threshold})`,
            data: keys.map(() => threshold),
            type: 'line',
            borderColor: PALETTE.danger,
            borderDash: [6, 4],
            pointRadius: 0,
            borderWidth: 1.5,
          },
        ],
      },
      options: themed({
        scales: { y: { min: 0, max: 1, title: { display: true, text: 'Adjusted R²' } } },
      }),
    });
  }
}
