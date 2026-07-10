document.addEventListener('DOMContentLoaded', () => {
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdVal = document.getElementById('threshold-val');
    const form = document.getElementById('routing-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnLoader = document.getElementById('btn-loader');
    const promptInput = document.getElementById('prompt-input');

    // Output DOM elements
    const placeholder = document.getElementById('output-placeholder');
    const metricsContainer = document.getElementById('metrics-container');
    const terminalContainer = document.getElementById('terminal-container');
    const routeProvider = document.getElementById('route-provider');
    const routeModel = document.getElementById('route-model');
    const routeScore = document.getElementById('route-score');
    const scoreProgress = document.getElementById('score-progress');
    const routeReason = document.getElementById('route-reason');
    const routeLatency = document.getElementById('route-latency');
    const routeFallback = document.getElementById('route-fallback');
    const completionText = document.getElementById('completion-text');

    // Config DOM elements
    const localModelName = document.getElementById('local-model-name');
    const remoteModelName = document.getElementById('remote-model-name');

    // Slider synchronisation
    thresholdSlider.addEventListener('input', (e) => {
        thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Load initial server configuration
    async function loadConfig() {
        try {
            const res = await fetch('/api/config');
            if (res.ok) {
                const config = await res.json();
                localModelName.textContent = config.local_model;
                remoteModelName.textContent = config.remote_model;
                thresholdSlider.value = config.threshold;
                thresholdVal.textContent = parseFloat(config.threshold).toFixed(2);
            }
        } catch (err) {
            console.error('Failed to load system config:', err);
            localModelName.textContent = 'Connection Error';
            remoteModelName.textContent = 'Connection Error';
        }
    }

    // Handle prompt submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const prompt = promptInput.value.strip ? promptInput.value.strip() : promptInput.value;
        const threshold = parseFloat(thresholdSlider.value);

        if (!prompt) return;

        // Toggle loading states
        submitBtn.disabled = true;
        btnLoader.style.display = 'block';
        
        try {
            const res = await fetch('/api/route', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, threshold })
            });

            if (res.ok) {
                const data = await res.json();
                
                // Hide placeholder, display dashboards
                placeholder.classList.add('hidden');
                metricsContainer.classList.remove('hidden');
                terminalContainer.classList.remove('hidden');

                // Render metrics values
                routeProvider.textContent = data.provider_used.toUpperCase();
                
                // Highlight local vs remote colors
                routeProvider.className = 'metric-value';
                if (data.provider_used === 'local') {
                    routeProvider.classList.add('local-color');
                } else {
                    routeProvider.classList.add('remote-color');
                }

                routeModel.textContent = data.model_used;
                routeScore.textContent = data.complexity_score.toFixed(2);
                scoreProgress.style.width = `${data.complexity_score * 100}%`;
                routeReason.textContent = data.routing_reason;
                routeLatency.textContent = `${data.latency_sec.toFixed(3)}s`;

                // Handle fallback metrics text
                if (data.fallback_used) {
                    routeFallback.textContent = 'Escalated Fallback';
                    routeFallback.className = 'metric-footer fallback-color';
                } else {
                    routeFallback.textContent = 'No Fallback';
                    routeFallback.className = 'metric-footer';
                }

                // Render LLM completion text
                completionText.textContent = data.response_content;
            } else {
                const errText = await res.text();
                alert(`Error routing request: ${errText}`);
            }
        } catch (err) {
            console.error('API request failed:', err);
            alert('Failed to connect to router API server.');
        } finally {
            submitBtn.disabled = false;
            btnLoader.style.display = 'none';
        }
    });

    loadConfig();
});
