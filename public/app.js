document.addEventListener('DOMContentLoaded', () => {
  const slider = document.getElementById('epsilonSlider');
  const epsValue = document.getElementById('epsValue');
  const accuracyDisplay = document.getElementById('accuracyDisplay');
  const statusMsg = document.getElementById('statusMsg');
  const advPred = document.getElementById('advPred');
  const root = document.documentElement;

  let debounceTimer = null;

  // 1. Immediate UI Feedback Loop (Visual Noise Update)
  slider.addEventListener('input', (e) => {
    const eps = parseFloat(e.target.value);
    epsValue.textContent = eps.toFixed(2);
    
    // Smoothly scale CSS variables without triggering server calls
    const noiseOpacity = (eps / 0.3) * 0.85; 
    root.style.setProperty('--pert-level', `${noiseOpacity}`);

    // Update frontend prediction label heuristics
    updatePredictionLabel(eps);

    // 2. Debounced API Request Trigger (Executes 300ms after user stops moving slider)
    clearTimeout(debounceTimer);
    statusMsg.textContent = "Computing perturbation matrix...";
    
    debounceTimer = setTimeout(() => {
      fetchEvaluationMetrics(eps);
    }, 300);
  });

  // 3. Async API Handler to Serverless Backend
  async function fetchEvaluationMetrics(epsilon) {
    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: "demo_model",
          epsilon: epsilon
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      const data = result.data;

      // Update UI with Server Calculation
      accuracyDisplay.textContent = `${data.robust_accuracy}%`;
      statusMsg.textContent = `Evaluated on ${data.perturbed_samples_count} samples. Accuracy drop: ${data.accuracy_drop}%`;

      if (data.robust_accuracy < 50.0) {
        accuracyDisplay.classList.add('drop');
      } else {
        accuracyDisplay.classList.remove('drop');
      }
    } catch (error) {
      console.error("Evaluation request failed:", error);
      statusMsg.textContent = "Error executing ART evaluation.";
    }
  }

  function updatePredictionLabel(eps) {
    if (eps > 0.15) {
      advPred.textContent = "Ostrich (84%)";
      advPred.style.color = "var(--accent-alert)";
    } else if (eps > 0.05) {
      advPred.textContent = "Dog (52%)";
      advPred.style.color = "var(--accent-warn)";
    } else {
      advPred.textContent = "Pug (98%)";
      advPred.style.color = "var(--text-main)";
    }
  }
});