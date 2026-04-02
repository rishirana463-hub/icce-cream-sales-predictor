/**
 * Ice Cream Sales Predictor - Frontend Script
 * Handles form submission and API communication
 */

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", function () {
  loadModelInfo();
  setupFormListener();
});

/**
 * Load and display model information
 */
function loadModelInfo() {
  fetch("/model-info")
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        displayModelInfo(data);
      } else {
        displayModelError(data.error);
      }
    })
    .catch((error) => {
      console.error("Error loading model info:", error);
      displayModelError("Failed to load model information");
    });
}

/**
 * Display model information on the page
 */
function displayModelInfo(data) {
  const infoContent = document.getElementById("model-info-content");

  const html = `
        <p><strong>Algorithm:</strong> ${data.algorithm}</p>
        <p><strong>Input Feature:</strong> ${data.feature}</p>
        <p><strong>Output Target:</strong> ${data.target}</p>
        <p><strong>Coefficient (Slope):</strong> ${data.coefficient.toFixed(4)}</p>
        <p><strong>Intercept:</strong> ${data.intercept.toFixed(4)}</p>
        <p class="equation"><strong>Equation:</strong><br>${data.equation}</p>
    `;

  infoContent.innerHTML = html;
}

/**
 * Display model loading error
 */
function displayModelError(error) {
  const infoContent = document.getElementById("model-info-content");
  infoContent.innerHTML = `<p style="color: #d32f2f;"><strong>Error:</strong> ${error}</p>`;
}

/**
 * Setup form submission listener
 */
function setupFormListener() {
  const form = document.getElementById("prediction-form");

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    makePrediction();
  });
}

/**
 * Make prediction API call
 */
function makePrediction() {
  // Get temperature input
  const temperatureInput = document.getElementById("temperature");
  const temperature = parseFloat(temperatureInput.value);

  // Validate input
  if (isNaN(temperature)) {
    showError("Please enter a valid temperature value");
    return;
  }

  if (temperature < -50 || temperature > 150) {
    showError("Temperature must be between -50°F and 150°F");
    return;
  }

  // Disable submit button during request
  const submitBtn = document.querySelector(".btn");
  const btnText = submitBtn.querySelector(".btn-text");
  submitBtn.disabled = true;
  btnText.textContent = "Predicting...";

  // Make API call
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ temperature: temperature }),
  })
    .then((response) => response.json())
    .then((data) => {
      // Re-enable submit button
      submitBtn.disabled = false;
      btnText.textContent = "Predict Sales";

      if (data.success) {
        showResult(data);
      } else {
        showError(data.error);
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      submitBtn.disabled = false;
      btnText.textContent = "Predict Sales";
      showError("An error occurred while making the prediction");
    });
}

/**
 * Display prediction result
 */
function showResult(data) {
  const resultContainer = document.getElementById("result-container");
  const resultContent = document.getElementById("result-content");
  const errorContainer = document.getElementById("error-container");

  // Hide error container
  errorContainer.classList.add("hidden");

  // Display result
  const html = `
    <div class="result-message">${data.message}</div>
    <div class="result-value">${data.predicted_sales.toFixed(0)}</div>
    <div style="margin-top: 10px; font-size: 0.95em; opacity: 0.9;">
      At <strong>${data.temperature}°F</strong>, predict approximately <strong>${data.predicted_sales.toFixed(0)}</strong> units
    </div>
  `;

  resultContent.innerHTML = html;
  resultContainer.classList.remove("hidden");
  resultContainer.classList.add("show");
}

/**
 * Display error message
 */
function showError(message) {
  const errorContainer = document.getElementById("error-container");
  const errorContent = document.getElementById("error-content");
  const resultContainer = document.getElementById("result-container");

  // Hide result container
  resultContainer.classList.add("hidden");
  resultContainer.classList.remove("show");

  // Display error
  errorContent.innerHTML = message;
  errorContainer.classList.remove("hidden");
  errorContainer.classList.add("show");
}
