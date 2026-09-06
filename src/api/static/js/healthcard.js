import { payload, themed, makeChart, PALETTE } from './charts.js';

const data = payload('card-data');
if (data) {
  // --- six-pillar radar ---------------------------------------------------
  if (data.pillars?.length) {
    makeChart('pillarRadar', {
      type: 'radar',
      data: {
        labels: data.pillars.map((p) => p.label),
        datasets: [{
          label: 'Pillar score',
          data: data.pillars.map((p) => p.score),
          borderColor: PALETTE.primary,
          backgroundColor: 'rgba(13,110,253,0.18)',
          pointBackgroundColor: PALETTE.primary,
        }],
      },
      options: themed({
        plugins: { legend: { display: false } },
        scales: { r: { suggestedMin: 0, suggestedMax: 100, ticks: { stepSize: 25 } } },
      }),
    });
  }

  // --- SHAP drivers -------------------------------------------------------
  // Signed horizontal bars: red raises default probability, green lowers it.
  if (data.shap?.length) {
    const sorted = [...data.shap].sort((a, b) => Math.abs(b.shap_pp) - Math.abs(a.shap_pp));
    makeChart('shapWaterfall', {
      type: 'bar',
      data: {
        labels: sorted.map((s) => s.label),
        datasets: [{
          label: 'Effect on PD (percentage points)',
          data: sorted.map((s) => s.shap_pp),
          backgroundColor: sorted.map((s) => (s.shap_pp > 0 ? PALETTE.danger : PALETTE.success)),
          borderRadius: 3,
        }],
      },
      options: themed({
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const v = ctx.parsed.x;
                return `${v > 0 ? 'raises' : 'lowers'} PD by ${Math.abs(v).toFixed(2)} pp`;
              },
            },
          },
        },
        scales: { x: { title: { display: true, text: 'percentage points of default probability' } } },
      }),
    });
  }
}
