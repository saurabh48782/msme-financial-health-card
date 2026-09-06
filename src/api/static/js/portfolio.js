import { payload, themed, makeChart, SERIES, BAND_COLOURS, GRADE_COLOURS, percent } from './charts.js';

const data = payload('portfolio-data');
if (data) {
  makeChart('fhsHistogram', {
    type: 'bar',
    data: {
      labels: data.histogram.map((b) => b.label),
      datasets: [{
        label: 'Firms',
        data: data.histogram.map((b) => b.count),
        backgroundColor: SERIES[0],
        borderRadius: 3,
      }],
    },
    options: themed({
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Financial Health Score' } },
        y: { beginAtZero: true, title: { display: true, text: 'Firms' } },
      },
    }),
  });

  makeChart('bandMix', {
    type: 'doughnut',
    data: {
      labels: data.bands.map((b) => b.label),
      datasets: [{
        data: data.bands.map((b) => b.count),
        backgroundColor: data.bands.map((b) => BAND_COLOURS[b.label] || SERIES[0]),
      }],
    },
    options: themed({ cutout: '58%' }),
  });

  makeChart('gradeMix', {
    type: 'bar',
    data: {
      labels: data.grades.map((b) => b.label),
      datasets: [{
        label: 'Firms',
        data: data.grades.map((b) => b.count),
        backgroundColor: data.grades.map((b) => GRADE_COLOURS[b.label] || SERIES[0]),
        borderRadius: 3,
      }],
    },
    options: themed({
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } },
    }),
  });

  // Approval rate is the point of this chart, so the axis is forced to a band
  // around the observed rates — autoscaling from zero would flatten a 2-point
  // gap into a straight line and hide exactly what is being claimed.
  const rates = data.segments.map((s) => s.approval_rate);
  const spread = Math.max(...rates) - Math.min(...rates);
  makeChart('segmentLift', {
    type: 'bar',
    data: {
      labels: data.segments.map((s) => s.segment),
      datasets: [{
        label: 'Approval rate',
        data: rates,
        backgroundColor: SERIES.slice(0, data.segments.length),
        borderRadius: 3,
      }],
    },
    options: themed({
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => percent(ctx.parsed.y) } },
      },
      scales: {
        y: {
          min: Math.max(0, Math.min(...rates) - Math.max(spread, 0.05)),
          max: Math.min(1, Math.max(...rates) + Math.max(spread, 0.05)),
          ticks: { callback: (v) => percent(v) },
          title: { display: true, text: 'Approval rate' },
        },
      },
    }),
  });

  makeChart('industryMix', {
    type: 'bar',
    data: {
      labels: data.industries.map((b) => b.label),
      datasets: [{
        label: 'Firms',
        data: data.industries.map((b) => b.count),
        backgroundColor: SERIES[1],
        borderRadius: 3,
      }],
    },
    options: themed({
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true } },
    }),
  });
}
