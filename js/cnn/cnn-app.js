/**
 * CNN App Controller
 * Orchestrates the CNN tab - dataset loading, model building, training, visualization
 */

import { DatasetLoader } from './dataset-loader.js';
import { CNNModelBuilder, LAYER_TYPES, ACTIVATIONS, OPTIMIZERS } from './cnn-model.js';
import { CNNVisualizer } from './cnn-viz.js';
import { CNNExplainer } from './cnn-explainer.js';

export class CNNApp {
    constructor() {
        this.datasetLoader = new DatasetLoader();
        this.modelBuilder = new CNNModelBuilder();
        this.visualizer = null;
        this.explainer = null;

        // Training state
        this.isTraining = false;
        this.isPaused = false;
        this.epoch = 0;
        this.step = 0;
        this.batchSize = 32;
        this.trainHistory = { loss: [], accuracy: [] };
        this.testHistory = { loss: [], accuracy: [] };

        // UI elements
        this.elements = {};

        // Animation frame for training loop
        this.trainingFrame = null;
        this.vizUpdateInterval = 10;  // Update viz every N steps
    }

    /**
     * Initialize the CNN app
     */
    async init() {
        console.log('Initializing CNN App...');

        // Get DOM elements
        this.cacheElements();

        // Initialize TensorFlow.js backend (GPU by default)
        await this.initBackend();

        // Setup visualizer
        const archCanvas = document.getElementById('cnn-arch-canvas');
        const featureCanvas = document.getElementById('cnn-feature-canvas');
        if (archCanvas && featureCanvas) {
            this.visualizer = new CNNVisualizer(archCanvas, featureCanvas);
            this.visualizer.resize();
        }

        // Setup explainer
        this.explainer = new CNNExplainer(this.modelBuilder, this.datasetLoader);
        this.explainer.init();

        // Setup event listeners
        this.setupEventListeners();

        // Setup default layer configuration (but don't build yet)
        this.modelBuilder.loadDefault();
        this.renderLayersList();

        // Load default dataset (this will also build the model with correct input shape)
        await this.loadDataset('mnist');

        console.log('CNN App initialized');
    }

    /**
     * Initialize TensorFlow.js backend
     * Tries WebGL (GPU) first, falls back to CPU if needed
     */
    async initBackend() {
        const preferredBackend = this.elements.backendSelect?.value || 'webgl';
        await this.setBackend(preferredBackend);
    }

    /**
     * Set TensorFlow.js backend
     * @param {string} backendName - 'webgl', 'webgpu', or 'cpu'
     */
    async setBackend(backendName) {
        // Update status to loading
        this.updateBackendStatus('loading');

        try {
            // Try to set the requested backend
            const success = await tf.setBackend(backendName);

            if (!success) {
                throw new Error(`Failed to set backend: ${backendName}`);
            }

            // Wait for backend to be ready
            await tf.ready();

            // Get actual backend name (might differ from requested)
            const actualBackend = tf.getBackend();
            console.log(`TensorFlow.js backend: ${actualBackend}`);

            // Update UI to show actual backend
            this.updateBackendStatus(actualBackend);
            this.updateBackendSelect(actualBackend);

            return actualBackend;
        } catch (error) {
            console.warn(`Backend ${backendName} not available:`, error.message);

            // Fall back to WebGL, then CPU
            if (backendName !== 'webgl') {
                console.log('Falling back to WebGL...');
                return this.setBackend('webgl');
            } else if (backendName !== 'cpu') {
                console.log('Falling back to CPU...');
                return this.setBackend('cpu');
            }

            // Update status
            const actualBackend = tf.getBackend() || 'cpu';
            this.updateBackendStatus(actualBackend);
            return actualBackend;
        }
    }

    /**
     * Update backend status indicator
     * @param {string} status - Backend name or 'loading'
     */
    updateBackendStatus(status) {
        const statusEl = this.elements.backendStatus;
        if (!statusEl) return;

        // Remove all status classes
        statusEl.classList.remove('backend-status--gpu', 'backend-status--cpu', 'backend-status--webgpu', 'backend-status--loading');

        // Add appropriate class
        if (status === 'loading') {
            statusEl.classList.add('backend-status--loading');
            statusEl.title = 'Loading...';
        } else if (status === 'webgl') {
            statusEl.classList.add('backend-status--gpu');
            statusEl.title = 'WebGL (GPU) - Active';
        } else if (status === 'webgpu') {
            statusEl.classList.add('backend-status--webgpu');
            statusEl.title = 'WebGPU - Active';
        } else {
            statusEl.classList.add('backend-status--cpu');
            statusEl.title = 'CPU - Active';
        }
    }

    /**
     * Update backend select dropdown to match actual backend
     * @param {string} backend - Actual backend name
     */
    updateBackendSelect(backend) {
        const selectEl = this.elements.backendSelect;
        if (!selectEl) return;

        // Map backend names to option values
        const optionValue = backend === 'webgl' ? 'webgl' :
                          backend === 'webgpu' ? 'webgpu' : 'cpu';

        selectEl.value = optionValue;
    }

    /**
     * Cache DOM element references
     */
    cacheElements() {
        this.elements = {
            // Dataset
            datasetSelect: document.getElementById('cnn-dataset-select'),
            loadingProgress: document.getElementById('cnn-loading-progress'),
            loadingBar: document.getElementById('cnn-loading-bar'),

            // Layer builder
            layersList: document.getElementById('cnn-layers-list'),
            addConvBtn: document.getElementById('add-conv-btn'),
            addPoolBtn: document.getElementById('add-pool-btn'),
            addDenseBtn: document.getElementById('add-dense-btn'),
            addDropoutBtn: document.getElementById('add-dropout-btn'),
            addBatchNormBtn: document.getElementById('add-batchnorm-btn'),

            // Training controls
            trainBtn: document.getElementById('cnn-train-btn'),
            resetBtn: document.getElementById('cnn-reset-btn'),
            stepBtn: document.getElementById('cnn-step-btn'),
            learningRateInput: document.getElementById('cnn-learning-rate'),
            optimizerSelect: document.getElementById('cnn-optimizer'),
            batchSizeInput: document.getElementById('cnn-batch-size'),

            // Metrics display
            epochDisplay: document.getElementById('cnn-epoch'),
            lossDisplay: document.getElementById('cnn-loss'),
            accuracyDisplay: document.getElementById('cnn-accuracy'),
            testAccuracyDisplay: document.getElementById('cnn-test-accuracy'),

            // Preview
            previewCanvas: document.getElementById('cnn-preview-canvas'),
            predictionBars: document.getElementById('cnn-prediction-bars'),

            // Canvases
            archCanvas: document.getElementById('cnn-arch-canvas'),
            featureCanvas: document.getElementById('cnn-feature-canvas'),

            // Backend selection
            backendSelect: document.getElementById('cnn-backend-select'),
            backendStatus: document.getElementById('cnn-backend-status'),

            // Explainer
            explainBtn: document.getElementById('cnn-explain-btn')
        };
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Dataset selection
        this.elements.datasetSelect?.addEventListener('change', (e) => {
            this.loadDataset(e.target.value);
        });

        // Layer buttons
        this.elements.addConvBtn?.addEventListener('click', () => this.addLayer('conv2d'));
        this.elements.addPoolBtn?.addEventListener('click', () => this.addLayer('maxPooling2d'));
        this.elements.addDenseBtn?.addEventListener('click', () => this.addLayer('dense'));
        this.elements.addDropoutBtn?.addEventListener('click', () => this.addLayer('dropout'));
        this.elements.addBatchNormBtn?.addEventListener('click', () => this.addLayer('batchNormalization'));

        // Training controls
        this.elements.trainBtn?.addEventListener('click', () => this.toggleTraining());
        this.elements.resetBtn?.addEventListener('click', () => this.reset());
        this.elements.stepBtn?.addEventListener('click', () => this.trainStep());

        // Hyperparameters
        this.elements.learningRateInput?.addEventListener('change', (e) => {
            this.modelBuilder.setOptimizer(
                this.elements.optimizerSelect?.value || 'adam',
                parseFloat(e.target.value)
            );
            this.rebuildModel();
        });

        this.elements.optimizerSelect?.addEventListener('change', (e) => {
            this.modelBuilder.setOptimizer(
                e.target.value,
                parseFloat(this.elements.learningRateInput?.value || 0.001)
            );
            this.rebuildModel();
        });

        this.elements.batchSizeInput?.addEventListener('change', (e) => {
            this.batchSize = parseInt(e.target.value) || 32;
        });

        // Backend selection
        this.elements.backendSelect?.addEventListener('change', async (e) => {
            const newBackend = e.target.value;
            console.log(`Switching backend to: ${newBackend}`);

            // Stop any ongoing training
            if (this.isTraining) {
                this.stopTraining();
            }

            // Dispose current model before switching backend
            this.modelBuilder.dispose();

            // Switch backend
            await this.setBackend(newBackend);

            // Rebuild model with new backend
            await this.rebuildModel();
        });

        // Explain button - open CNN explainer
        this.elements.explainBtn?.addEventListener('click', () => {
            this.openExplainer();
        });

        // Architecture canvas click - select layer
        this.elements.archCanvas?.addEventListener('click', (e) => {
            const rect = this.elements.archCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const layerIndex = this.visualizer?.getLayerAtPosition(x, y);
            if (layerIndex !== null && layerIndex >= 0) {
                this.selectLayer(layerIndex);
            }
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.visualizer?.resize();
        });
    }

    /**
     * Load a dataset
     * @param {string} datasetId - 'mnist', 'fashion', or 'cifar10'
     */
    async loadDataset(datasetId) {
        console.log(`Loading dataset: ${datasetId}`);

        // Show loading progress
        if (this.elements.loadingProgress) {
            this.elements.loadingProgress.style.display = 'block';
        }

        try {
            await this.datasetLoader.load(datasetId, (progress) => {
                if (this.elements.loadingBar) {
                    this.elements.loadingBar.style.width = `${progress}%`;
                }
            });

            // Update model input shape
            const inputShape = this.datasetLoader.getInputShape();
            const config = this.datasetLoader.getConfig();
            this.modelBuilder.setInputShape(inputShape);
            this.modelBuilder.setNumClasses(config.numClasses);

            // Rebuild model with new input shape
            this.rebuildModel();

            // Show sample
            this.showRandomSample();

            console.log(`Dataset ${datasetId} loaded successfully`);
        } catch (error) {
            console.error('Failed to load dataset:', error);
        } finally {
            if (this.elements.loadingProgress) {
                this.elements.loadingProgress.style.display = 'none';
            }
        }
    }

    /**
     * Build default CNN model (resets to default layers and rebuilds)
     */
    buildDefaultModel() {
        this.modelBuilder.loadDefault();
        this.renderLayersList();
        if (this.datasetLoader.isLoaded) {
            this.rebuildModel();
        }
    }

    /**
     * Add a layer to the model
     * @param {string} type - Layer type
     */
    addLayer(type) {
        // Insert before the last dense layer (output) if it exists
        const layers = this.modelBuilder.layers;
        let insertIndex = layers.length;

        // Find the output layer (last dense with softmax)
        for (let i = layers.length - 1; i >= 0; i--) {
            if (layers[i].type === 'dense' && layers[i].config.activation === 'softmax') {
                insertIndex = i;
                break;
            }
        }

        // Add flatten before dense layers if needed
        if ((type === 'dense' || type === 'dropout') && insertIndex > 0) {
            const prevLayer = layers[insertIndex - 1];
            if (prevLayer && !['flatten', 'dense', 'dropout'].includes(prevLayer.type)) {
                this.modelBuilder.layers.splice(insertIndex, 0, {
                    id: Date.now() + Math.random(),
                    type: 'flatten',
                    config: {}
                });
                insertIndex++;
            }
        }

        // Create new layer
        const layerType = LAYER_TYPES[type];
        const newLayer = {
            id: Date.now() + Math.random(),
            type: type,
            config: { ...layerType.defaults }
        };

        // Insert layer
        this.modelBuilder.layers.splice(insertIndex, 0, newLayer);

        this.renderLayersList();
        this.rebuildModel();
        this.resetTraining();
    }

    /**
     * Remove a layer
     * @param {number} index - Layer index
     */
    removeLayer(index) {
        if (this.modelBuilder.layers.length <= 2) {
            alert('Model needs at least 2 layers');
            return;
        }
        this.modelBuilder.removeLayer(index);
        this.renderLayersList();
        this.rebuildModel();
        this.resetTraining();
    }

    /**
     * Update layer configuration
     * @param {number} index - Layer index
     * @param {string} param - Parameter name
     * @param {any} value - New value
     */
    updateLayerConfig(index, param, value) {
        const layer = this.modelBuilder.layers[index];
        if (layer) {
            layer.config[param] = value;
            this.rebuildModel();
            this.resetTraining();
        }
    }

    /**
     * Rebuild the TensorFlow.js model
     */
    rebuildModel() {
        try {
            this.modelBuilder.build();
            this.visualizer?.setModel(
                this.modelBuilder.model,
                this.modelBuilder.layers
            );
            this.updateModelSummary();
        } catch (error) {
            console.error('Failed to build model:', error);
        }
    }

    /**
     * Update model summary display
     */
    updateModelSummary() {
        const summary = this.modelBuilder.getSummary();
        if (summary) {
            console.log(`Model: ${summary.totalParams.toLocaleString()} parameters`);
        }
    }

    /**
     * Render layers list in sidebar - uses safe DOM methods
     */
    renderLayersList() {
        if (!this.elements.layersList) return;

        const layers = this.modelBuilder.layers;
        this.elements.layersList.textContent = '';  // Clear safely

        layers.forEach((layer, index) => {
            const card = this.createLayerCard(layer, index);
            this.elements.layersList.appendChild(card);
        });
    }

    /**
     * Create a layer card element using safe DOM methods
     */
    createLayerCard(layer, index) {
        const card = document.createElement('div');
        card.className = 'layer-card';
        card.dataset.index = index;

        const layerType = LAYER_TYPES[layer.type];
        card.style.borderLeftColor = layerType?.color || '#64748b';

        // Header
        const header = document.createElement('div');
        header.className = 'layer-card__header';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'layer-card__name';
        nameSpan.textContent = layerType?.name || layer.type;

        const removeBtn = document.createElement('button');
        removeBtn.className = 'layer-card__remove';
        removeBtn.title = 'Remove layer';
        removeBtn.textContent = '\u00d7';
        removeBtn.addEventListener('click', () => this.removeLayer(index));

        header.appendChild(nameSpan);
        header.appendChild(removeBtn);
        card.appendChild(header);

        // Config section
        const config = document.createElement('div');
        config.className = 'layer-card__config';

        switch (layer.type) {
            case 'conv2d':
                config.appendChild(this.createSelectControl('Filters', 'filters', [8, 16, 32, 64, 128], layer.config.filters, index));
                config.appendChild(this.createSelectControl('Kernel', 'kernelSize', [3, 5, 7], layer.config.kernelSize, index, (v) => `${v}x${v}`));
                config.appendChild(this.createSelectControl('Activation', 'activation', ACTIVATIONS, layer.config.activation, index));
                break;

            case 'maxPooling2d':
            case 'avgPooling2d':
                config.appendChild(this.createSelectControl('Pool Size', 'poolSize', [2, 3], layer.config.poolSize, index, (v) => `${v}x${v}`));
                break;

            case 'dense':
                config.appendChild(this.createSelectControl('Units', 'units', [32, 64, 128, 256, 512], layer.config.units, index));
                config.appendChild(this.createSelectControl('Activation', 'activation', ACTIVATIONS, layer.config.activation, index));
                break;

            case 'dropout':
                config.appendChild(this.createRangeControl('Rate', 'rate', 0.1, 0.5, 0.05, layer.config.rate, index));
                break;

            case 'flatten':
            case 'batchNormalization':
                const noConfig = document.createElement('span');
                noConfig.className = 'layer-card__no-config';
                noConfig.textContent = 'No configuration';
                config.appendChild(noConfig);
                break;
        }

        card.appendChild(config);
        return card;
    }

    /**
     * Create a select control for layer config
     */
    createSelectControl(labelText, param, options, currentValue, layerIndex, formatFn = null) {
        const label = document.createElement('label');
        label.textContent = labelText + ': ';

        const select = document.createElement('select');
        select.dataset.param = param;

        options.forEach(opt => {
            const option = document.createElement('option');
            option.value = opt;
            option.textContent = formatFn ? formatFn(opt) : opt;
            if (opt === currentValue || String(opt) === String(currentValue)) {
                option.selected = true;
            }
            select.appendChild(option);
        });

        select.addEventListener('change', (e) => {
            let value = e.target.value;
            if (!isNaN(parseInt(value))) {
                value = parseInt(value);
            }
            this.updateLayerConfig(layerIndex, param, value);
        });

        label.appendChild(select);
        return label;
    }

    /**
     * Create a range control for layer config
     */
    createRangeControl(labelText, param, min, max, step, currentValue, layerIndex) {
        const label = document.createElement('label');
        label.textContent = labelText + ': ';

        const range = document.createElement('input');
        range.type = 'range';
        range.dataset.param = param;
        range.min = min;
        range.max = max;
        range.step = step;
        range.value = currentValue;

        const valueSpan = document.createElement('span');
        valueSpan.className = 'rate-value';
        valueSpan.textContent = `${(currentValue * 100).toFixed(0)}%`;

        range.addEventListener('change', (e) => {
            const value = parseFloat(e.target.value);
            valueSpan.textContent = `${(value * 100).toFixed(0)}%`;
            this.updateLayerConfig(layerIndex, param, value);
        });

        label.appendChild(range);
        label.appendChild(valueSpan);
        return label;
    }

    /**
     * Select a layer for visualization
     */
    selectLayer(index) {
        this.visualizer?.setSelectedLayer(index);
        this.updateFeatureMapViz();
    }

    /**
     * Toggle training on/off
     */
    toggleTraining() {
        if (this.isTraining) {
            this.stopTraining();
        } else {
            this.startTraining();
        }
    }

    /**
     * Start training
     */
    startTraining() {
        if (!this.datasetLoader.isLoaded) {
            alert('Please wait for dataset to load');
            return;
        }
        if (!this.modelBuilder.model) {
            alert('Please build a model first');
            return;
        }

        this.isTraining = true;
        this.isPaused = false;

        this.updateTrainButtonState(true);
        this.trainingLoop();
    }

    /**
     * Stop training
     */
    stopTraining() {
        this.isTraining = false;
        if (this.trainingFrame) {
            cancelAnimationFrame(this.trainingFrame);
            this.trainingFrame = null;
        }

        this.updateTrainButtonState(false);
    }

    /**
     * Update train button visual state
     * @param {boolean} isTraining - Whether currently training
     */
    updateTrainButtonState(isTraining) {
        const btn = this.elements.trainBtn;
        if (!btn) return;

        const playIcon = btn.querySelector('.icon-play');
        const pauseIcon = btn.querySelector('.icon-pause');
        const btnText = btn.querySelector('.btn-text');

        if (isTraining) {
            // Show pause icon, hide play icon
            if (playIcon) playIcon.style.display = 'none';
            if (pauseIcon) pauseIcon.style.display = 'block';
            if (btnText) btnText.textContent = 'Stop';
            btn.classList.add('btn--danger');
            btn.classList.remove('btn--primary');
            btn.title = 'Stop Training';
        } else {
            // Show play icon, hide pause icon
            if (playIcon) playIcon.style.display = 'block';
            if (pauseIcon) pauseIcon.style.display = 'none';
            if (btnText) btnText.textContent = 'Train';
            btn.classList.remove('btn--danger');
            btn.classList.add('btn--primary');
            btn.title = 'Train';
        }
    }

    /**
     * Training loop
     */
    async trainingLoop() {
        if (!this.isTraining || this.isPaused) {
            if (this.isTraining) {
                this.trainingFrame = requestAnimationFrame(() => this.trainingLoop());
            }
            return;
        }

        await this.trainStep();

        if (this.isTraining) {
            this.trainingFrame = requestAnimationFrame(() => this.trainingLoop());
        }
    }

    /**
     * Single training step
     */
    async trainStep() {
        if (!this.modelBuilder.model || !this.datasetLoader.isLoaded) return;

        const batch = this.datasetLoader.nextTrainBatch(this.batchSize);

        try {
            const result = await this.modelBuilder.trainOnBatch(batch);

            this.step++;
            if (this.step % Math.ceil(this.datasetLoader.datasetConfig.numTrain / this.batchSize) === 0) {
                this.epoch++;
            }

            // Update metrics display
            this.trainHistory.loss.push(result.loss);
            this.trainHistory.accuracy.push(result.accuracy);
            this.updateMetricsDisplay(result);

            // Update visualization periodically
            if (this.step % this.vizUpdateInterval === 0) {
                this.updateFeatureMapViz();
            }

            // Evaluate on test set periodically
            if (this.step % 50 === 0) {
                await this.evaluateOnTestSet();
            }
        } catch (error) {
            console.error('Training error:', error);
            this.stopTraining();
        } finally {
            // Dispose batch tensors
            batch.xs.dispose();
            batch.ys.dispose();
        }
    }

    /**
     * Evaluate on test set
     */
    async evaluateOnTestSet() {
        const testBatch = this.datasetLoader.nextTestBatch(100);
        try {
            const result = await this.modelBuilder.evaluate(testBatch);
            this.testHistory.loss.push(result.loss);
            this.testHistory.accuracy.push(result.accuracy);

            if (this.elements.testAccuracyDisplay) {
                this.elements.testAccuracyDisplay.textContent = `${(result.accuracy * 100).toFixed(1)}%`;
            }
        } finally {
            testBatch.xs.dispose();
            testBatch.ys.dispose();
        }
    }

    /**
     * Update metrics display
     */
    updateMetricsDisplay(result) {
        if (this.elements.epochDisplay) {
            this.elements.epochDisplay.textContent = this.epoch;
        }
        if (this.elements.lossDisplay) {
            this.elements.lossDisplay.textContent = result.loss.toFixed(4);
        }
        if (this.elements.accuracyDisplay) {
            this.elements.accuracyDisplay.textContent = `${(result.accuracy * 100).toFixed(1)}%`;
        }
    }

    /**
     * Update feature map visualization
     */
    async updateFeatureMapViz() {
        if (!this.visualizer || !this.modelBuilder.model) return;

        // Get a sample
        const sample = this.datasetLoader.getRandomSample(false);
        if (!sample) return;

        try {
            // Get activations at selected layer
            const activations = this.modelBuilder.getActivations(
                sample.image,
                this.visualizer.selectedLayerIndex
            );

            if (activations) {
                await this.visualizer.renderFeatureMaps(activations);
                activations.dispose();
            }

            // Also update preview and prediction
            this.updatePreview(sample);
        } finally {
            sample.image.dispose();
        }
    }

    /**
     * Update preview canvas with sample and prediction
     */
    async updatePreview(sample) {
        if (!this.elements.previewCanvas) return;

        const ctx = this.elements.previewCanvas.getContext('2d');
        const config = this.datasetLoader.getConfig();

        // Clear
        ctx.clearRect(0, 0, this.elements.previewCanvas.width, this.elements.previewCanvas.height);

        // Render input image
        this.visualizer?.renderInputImage(
            ctx, sample.imageData,
            config.imageSize, config.channels,
            10, 10, 80
        );

        // Get prediction
        if (this.modelBuilder.model) {
            const pred = this.modelBuilder.predict(sample.image);
            const probs = await pred.data();
            pred.dispose();

            // Show warning if model is untrained
            if (!this.modelBuilder.isTrained()) {
                ctx.fillStyle = '#f59e0b';
                ctx.font = 'bold 10px system-ui, sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText('⚠ Untrained', 100, 10);
                ctx.fillStyle = '#94a3b8';
                ctx.font = '9px system-ui, sans-serif';
                ctx.fillText('Train model for real predictions', 100, 22);
            }

            // Render prediction bars
            this.visualizer?.renderPredictionBars(
                ctx, probs, config.classNames,
                100, this.modelBuilder.isTrained() ? 10 : 35, 150, this.modelBuilder.isTrained() ? 80 : 55
            );
        }
    }

    /**
     * Show a random sample
     */
    showRandomSample() {
        if (!this.datasetLoader.isLoaded) return;

        const sample = this.datasetLoader.getRandomSample(false);
        if (sample) {
            this.updatePreview(sample);
            sample.image.dispose();
        }
    }

    /**
     * Reset training
     */
    resetTraining() {
        this.epoch = 0;
        this.step = 0;
        this.trainHistory = { loss: [], accuracy: [] };
        this.testHistory = { loss: [], accuracy: [] };
        this.updateMetricsDisplay({ loss: 0, accuracy: 0 });
        if (this.elements.testAccuracyDisplay) {
            this.elements.testAccuracyDisplay.textContent = '-';
        }
    }

    /**
     * Full reset - rebuild model and reset training
     */
    reset() {
        this.stopTraining();
        this.resetTraining();
        this.rebuildModel();
        this.showRandomSample();
    }

    /**
     * Open CNN Explainer modal
     */
    openExplainer() {
        if (!this.modelBuilder.model) {
            alert('Please build a model first!');
            return;
        }
        if (!this.datasetLoader.isLoaded) {
            alert('Please wait for dataset to load first!');
            return;
        }
        if (!this.modelBuilder.isTrained()) {
            alert('Please train the model first! Click "Train" and wait for at least one epoch to complete.');
            return;
        }
        this.explainer?.open();
    }

    /**
     * Clean up resources
     */
    dispose() {
        this.stopTraining();
        this.modelBuilder.dispose();
        this.visualizer?.dispose();
        this.explainer?.dispose();
    }
}

// Auto-initialize when DOM is ready
let cnnApp = null;

export function initCNNApp() {
    if (!cnnApp) {
        cnnApp = new CNNApp();
        cnnApp.init();
    }
    return cnnApp;
}

export function getCNNApp() {
    return cnnApp;
}
