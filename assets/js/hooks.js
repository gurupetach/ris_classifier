// Import Chart.js
import { Chart } from 'chart.js/auto'

let Hooks = {}

Hooks.TrainingChart = {
  mounted() {
    console.log("TrainingChart hook mounted");
    // Initialize the chart
    this.chart = new Chart(this.el, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Accuracy',
            data: [],
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderWidth: 2,
            tension: 0.1,
            fill: true
          },
          {
            label: 'Loss',
            data: [],
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderWidth: 2,
            tension: 0.1,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              padding: 20,
              usePointStyle: true,
              pointStyle: 'circle'
            }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              label: function(context) {
                let label = context.dataset.label || '';
                if (label) {
                  label += ': ';
                }
                if (context.parsed.y !== null) {
                  // Format accuracy as percentage, loss as decimal
                  if (label.includes('Accuracy')) {
                    label += (context.parsed.y * 100).toFixed(2) + '%';
                  } else {
                    label += context.parsed.y.toFixed(4);
                  }
                }
                return label;
              }
            }
          }
        },
        interaction: {
          intersect: false,
          mode: 'index'
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: 'Epoch',
              font: {
                size: 14
              }
            },
            grid: {
              display: false
            }
          },
          y: {
            display: true,
            beginAtZero: true,
            suggestedMax: 1,
            grid: {
              color: 'rgba(0, 0, 0, 0.1)'
            },
            ticks: {
              callback: function(value) {
                return (value * 100).toFixed(0) + '%';
              }
            }
          }
        },
        animation: {
          duration: 150
        }
      }
    });

    // Handle the metrics-updated event from LiveView
    this.handleEvent("metrics-updated", ({ epochs_history }) => {
      if (!epochs_history || epochs_history.length === 0) return;

      // Sort epochs history by epoch number to ensure correct order
      const sortedHistory = [...epochs_history].sort((a, b) => a.epoch - b.epoch);

      // Extract data for the chart
      const epochs = sortedHistory.map(m => m.epoch);
      const accuracy = sortedHistory.map(m => m.accuracy);
      const loss = sortedHistory.map(m => m.loss);

      // Update chart data
      this.chart.data.labels = epochs;
      this.chart.data.datasets[0].data = accuracy;
      this.chart.data.datasets[1].data = loss;

      // Update chart with animation
      this.chart.update('active');
    });
  },

  // Clean up when the element is removed
  destroyed() {
    if (this.chart) {
      this.chart.destroy();
    }
  },

  // Update chart if the element is updated
  updated() {
    if (this.chart) {
      this.chart.update('active');
    }
  }
}

export default Hooks;