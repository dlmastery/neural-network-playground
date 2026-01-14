/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Neural Network Playground - Main Application
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * A stunning interactive deep learning visualization experience
 */

import { Tensor } from './core/tensor.js';
import { NeuralNetwork } from './core/network.js';
import { DenseLayer } from './core/layer.js';
import { DatasetGenerator } from './datasets/generators.js';
import { ParticleSystem } from './visualization/particles.js';
import { NetworkVisualizer } from './visualization/network-viz.js';
import { BoundaryVisualizer } from './visualization/boundary-viz.js';

// Tab Controller - manages tab switching and lazy loading
class TabController {
    constructor() {
        this.activeTab = 'mlp';
        this.cnnApp = null;
        this.cnnInitialized = false;
        this.transformerApp = null;
        this.transformerInitialized = false;
        this.gnnApp = null;
        this.gnnInitialized = false;
        this.diffusionApp = null;
        this.diffusionInitialized = false;
        this.designSpacesApp = null;
        this.designSpacesInitialized = false;
        this.init();
    }

    init() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                this.switchTab(tab);
            });
        });
    }

    async switchTab(tab) {
        if (tab === this.activeTab) return;

        this.activeTab = tab;

        // Update button states
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('tab-btn--active', btn.dataset.tab === tab);
        });

        // Update panel visibility
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.toggle('tab-panel--active', panel.id === `${tab}-panel`);
        });

        // Lazy load CNN app when first accessed
        if (tab === 'cnn' && !this.cnnInitialized) {
            await this.initCNNApp();
        }

        // Lazy load Transformer app when first accessed
        if (tab === 'transformer' && !this.transformerInitialized) {
            await this.initTransformerApp();
        }

        // Lazy load GNN app when first accessed
        if (tab === 'gnn' && !this.gnnInitialized) {
            await this.initGNNApp();
        }

        // Lazy load Diffusion app when first accessed
        if (tab === 'diffusion' && !this.diffusionInitialized) {
            await this.initDiffusionApp();
        }

        // Lazy load Design Spaces app when first accessed
        if (tab === 'design-spaces' && !this.designSpacesInitialized) {
            await this.initDesignSpacesApp();
        }

        // Notify active app of tab switch
        if (tab === 'transformer' && this.transformerApp) {
            this.transformerApp.onActivate();
        }
        if (tab === 'gnn' && this.gnnApp) {
            this.gnnApp.onActivate();
        }

        // Trigger resize for visualizations to adapt
        window.dispatchEvent(new Event('resize'));
    }

    async initCNNApp() {
        try {
            // Wait for TensorFlow.js to be ready
            if (typeof tf === 'undefined') {
                console.error('TensorFlow.js not loaded');
                return;
            }

            console.log('Initializing CNN App with TensorFlow.js', tf.version.tfjs);

            // Dynamically import CNN app
            const { initCNNApp } = await import('./cnn/cnn-app.js');
            this.cnnApp = await initCNNApp();
            this.cnnInitialized = true;

            console.log('CNN App initialized successfully');
        } catch (error) {
            console.error('Failed to initialize CNN App:', error);
        }
    }

    async initTransformerApp() {
        try {
            console.log('Initializing Transformer App...');

            // Dynamically import Transformer app
            const { transformerApp } = await import('./transformer/transformer-app.js');
            await transformerApp.init();
            this.transformerApp = transformerApp;
            this.transformerInitialized = true;

            console.log('Transformer App initialized successfully');
        } catch (error) {
            console.error('Failed to initialize Transformer App:', error);
        }
    }

    async initGNNApp() {
        try {
            console.log('Initializing GNN App...');

            // Dynamically import GNN app
            const { GNNApp } = await import('./gnn/gnn-app.js');
            this.gnnApp = new GNNApp();
            await this.gnnApp.init();
            this.gnnInitialized = true;

            console.log('GNN App initialized successfully');
        } catch (error) {
            console.error('Failed to initialize GNN App:', error);
        }
    }

    async initDiffusionApp() {
        try {
            console.log('Initializing Diffusion App...');

            // Dynamically import Diffusion app and visualization
            const { diffusionApp } = await import('./diffusion/diffusion-app.js');
            const { diffusionViz } = await import('./diffusion/diffusion-viz.js');

            await diffusionApp.init();
            diffusionViz.init(
                document.getElementById('diffusion-grid-canvas'),
                document.getElementById('diffusion-timeline-canvas'),
                document.getElementById('diffusion-arch-mini-canvas')
            );

            this.diffusionApp = diffusionApp;
            this.diffusionInitialized = true;

            console.log('Diffusion App initialized successfully');
        } catch (error) {
            console.error('Failed to initialize Diffusion App:', error);
        }
    }

    async initDesignSpacesApp() {
        try {
            console.log('Initializing Design Spaces App...');

            // Dynamically import Design Spaces app
            const { designSpacesApp } = await import('./design-spaces/design-spaces-app.js');

            await designSpacesApp.init();

            this.designSpacesApp = designSpacesApp;
            this.designSpacesInitialized = true;

            console.log('Design Spaces App initialized successfully');
        } catch (error) {
            console.error('Failed to initialize Design Spaces App:', error);
        }
    }
}

class NeuralPlayground {
    constructor() {
        // State
        this.network = null;
        this.dataset = null;
        this.isTraining = false;
        this.trainingInterval = null;

        // Configuration
        this.config = {
            learningRate: 0.03,
            batchSize: 10,
            activation: 'tanh',
            regularization: 0,
            noise: 0,
            datasetType: 'circle',
            numPoints: 200
        };

        // Input features configuration
        this.inputFeatures = {
            x1: true,
            x2: true,
            x1sq: false,
            x2sq: false,
            x1x2: false,
            sinx1: false,
            sinx2: false
        };

        // Hidden layers configuration
        this.hiddenLayers = [
            { neurons: 8, activation: 'tanh' }
        ];

        // Metrics
        this.metrics = {
            epoch: 0,
            loss: 0,
            accuracy: 0.5,
            speed: 0
        };

        this.lossHistory = [];
        this.lastUpdateTime = Date.now();
        this.iterationCount = 0;

        this.init();
    }

    async init() {
        // Initialize visualizations
        this.initParticles();
        this.initNetworkViz();
        this.initBoundaryViz();
        this.initLossChart();

        // Initialize UI
        this.initControls();
        this.initSidebar();

        // Generate initial dataset and network
        this.generateDataset();
        this.buildNetwork();

        // Initial render
        this.updateVisualization();

        console.log('🧠 Neural Network Playground initialized');
    }

    initParticles() {
        const canvas = document.getElementById('particles-canvas');
        if (canvas) {
            this.particles = new ParticleSystem(canvas);
        }
    }

    initNetworkViz() {
        const canvas = document.getElementById('network-canvas');
        if (canvas) {
            this.networkViz = new NetworkVisualizer(canvas);
        }
    }

    initBoundaryViz() {
        const canvas = document.getElementById('boundary-canvas');
        if (canvas) {
            this.boundaryViz = new BoundaryVisualizer(canvas);
        }
    }

    initLossChart() {
        this.lossCanvas = document.getElementById('loss-chart');
        if (this.lossCanvas) {
            this.lossCtx = this.lossCanvas.getContext('2d');
        }
    }

    initControls() {
        // Dataset buttons
        const datasetGrid = document.getElementById('dataset-grid');
        if (datasetGrid) {
            datasetGrid.addEventListener('click', (e) => {
                const btn = e.target.closest('.dataset-btn');
                if (!btn) return;

                // Update active state
                datasetGrid.querySelectorAll('.dataset-btn').forEach(b =>
                    b.classList.remove('dataset-btn--active'));
                btn.classList.add('dataset-btn--active');

                // Generate new dataset
                this.config.datasetType = btn.dataset.dataset;
                this.generateDataset();
                this.resetTraining();
            });
        }

        // Train button
        const btnTrain = document.getElementById('btn-train');
        if (btnTrain) {
            btnTrain.addEventListener('click', () => this.toggleTraining());
        }

        // Step button
        const btnStep = document.getElementById('btn-step');
        if (btnStep) {
            btnStep.addEventListener('click', () => this.trainStep());
        }

        // Reset button
        const btnReset = document.getElementById('btn-reset');
        if (btnReset) {
            btnReset.addEventListener('click', () => this.resetTraining());
        }

        // Learning rate slider
        const lrSlider = document.getElementById('learning-rate');
        const lrValue = document.getElementById('lr-value');
        if (lrSlider && lrValue) {
            lrSlider.addEventListener('input', (e) => {
                this.config.learningRate = Math.pow(10, parseFloat(e.target.value));
                lrValue.textContent = this.config.learningRate.toFixed(3);
            });
        }

        // Activation select
        const activationSelect = document.getElementById('activation');
        if (activationSelect) {
            activationSelect.addEventListener('change', (e) => {
                this.config.activation = e.target.value;
                this.hiddenLayers.forEach(l => l.activation = e.target.value);
                this.buildNetwork();
                this.resetTraining();
            });
        }

        // Regularization slider
        const regSlider = document.getElementById('regularization');
        const regValue = document.getElementById('reg-value');
        if (regSlider && regValue) {
            regSlider.addEventListener('input', (e) => {
                this.config.regularization = parseFloat(e.target.value);
                regValue.textContent = this.config.regularization.toFixed(2);
            });
        }

        // Noise slider
        const noiseSlider = document.getElementById('noise');
        const noiseValue = document.getElementById('noise-value');
        if (noiseSlider && noiseValue) {
            noiseSlider.addEventListener('input', (e) => {
                this.config.noise = parseInt(e.target.value) / 100;
                noiseValue.textContent = `${e.target.value}%`;
                this.generateDataset();
            });
        }

        // Batch size radio buttons
        const batchRadios = document.querySelectorAll('input[name="batch-size"]');
        batchRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.config.batchSize = e.target.value === 'all' ? 'all' : parseInt(e.target.value);
            });
        });

        // Add layer button
        const btnAddLayer = document.getElementById('btn-add-layer');
        if (btnAddLayer) {
            btnAddLayer.addEventListener('click', () => this.addHiddenLayer());
        }

        // Input feature toggles
        const inputFeaturesContainer = document.getElementById('input-features');
        if (inputFeaturesContainer) {
            inputFeaturesContainer.addEventListener('click', (e) => {
                const toggle = e.target.closest('.feature-toggle');
                if (!toggle) return;

                const feature = toggle.dataset.feature;

                // Toggle the feature
                this.inputFeatures[feature] = !this.inputFeatures[feature];
                toggle.classList.toggle('feature-toggle--active', this.inputFeatures[feature]);

                // Ensure at least one feature is selected
                const activeCount = Object.values(this.inputFeatures).filter(v => v).length;
                if (activeCount === 0) {
                    this.inputFeatures[feature] = true;
                    toggle.classList.add('feature-toggle--active');
                    return;
                }

                // Update input count display
                this.updateInputCount();

                // Rebuild network with new input size
                this.buildNetwork();
                this.resetTraining();
            });
        }
    }

    initSidebar() {
        this.renderHiddenLayers();
        this.updateInputCount();
    }

    updateInputCount() {
        const inputCountEl = document.getElementById('input-count');
        if (inputCountEl) {
            const count = this.getActiveFeatureCount();
            inputCountEl.textContent = `${count} feature${count !== 1 ? 's' : ''}`;
        }
    }

    getActiveFeatureCount() {
        return Object.values(this.inputFeatures).filter(v => v).length;
    }

    /**
     * Transform raw x1, x2 coordinates into selected features
     */
    transformFeatures(x1, x2) {
        const features = [];

        if (this.inputFeatures.x1) features.push(x1);
        if (this.inputFeatures.x2) features.push(x2);
        if (this.inputFeatures.x1sq) features.push(x1 * x1);
        if (this.inputFeatures.x2sq) features.push(x2 * x2);
        if (this.inputFeatures.x1x2) features.push(x1 * x2);
        if (this.inputFeatures.sinx1) features.push(Math.sin(x1 * Math.PI));
        if (this.inputFeatures.sinx2) features.push(Math.sin(x2 * Math.PI));

        return features;
    }

    /**
     * Transform a Tensor of raw [x1, x2] points into selected features
     */
    transformDataset(X) {
        const numPoints = X.rows;
        const numFeatures = this.getActiveFeatureCount();
        const transformedData = new Float32Array(numPoints * numFeatures);

        for (let i = 0; i < numPoints; i++) {
            const x1 = X.get(i, 0);
            const x2 = X.get(i, 1);
            const features = this.transformFeatures(x1, x2);

            for (let j = 0; j < features.length; j++) {
                transformedData[i * numFeatures + j] = features[j];
            }
        }

        return new Tensor(numPoints, numFeatures, transformedData);
    }

    renderHiddenLayers() {
        const container = document.getElementById('hidden-layers');
        if (!container) return;

        container.innerHTML = '';

        this.hiddenLayers.forEach((layer, index) => {
            const layerEl = document.createElement('div');
            layerEl.className = 'hidden-layer animate-fade-in-up';
            layerEl.innerHTML = `
                <div class="hidden-layer__header">
                    <div class="hidden-layer__title">
                        <span class="hidden-layer__badge">Hidden ${index + 1}</span>
                    </div>
                    <button class="hidden-layer__remove" data-index="${index}" title="Remove layer">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
                <div class="hidden-layer__neurons">
                    <div class="hidden-layer__neuron-btns">
                        <button class="hidden-layer__neuron-btn" data-action="decrease" data-index="${index}">−</button>
                    </div>
                    <span class="hidden-layer__neuron-count">${layer.neurons}</span>
                    <div class="hidden-layer__neuron-btns">
                        <button class="hidden-layer__neuron-btn" data-action="increase" data-index="${index}">+</button>
                    </div>
                </div>
                <div class="neuron-row">
                    ${Array(Math.min(layer.neurons, 10)).fill(0).map(() =>
                        '<span class="neuron-dot"></span>'
                    ).join('')}
                    ${layer.neurons > 10 ? '<span class="neuron-dot" style="opacity: 0.5">...</span>' : ''}
                </div>
            `;

            container.appendChild(layerEl);
        });

        // Bind layer controls
        container.querySelectorAll('.hidden-layer__remove').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const index = parseInt(e.currentTarget.dataset.index);
                this.removeHiddenLayer(index);
            });
        });

        container.querySelectorAll('.hidden-layer__neuron-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const index = parseInt(e.currentTarget.dataset.index);
                const action = e.currentTarget.dataset.action;
                this.changeNeuronCount(index, action === 'increase' ? 1 : -1);
            });
        });

        this.updateNetworkSummary();
    }

    addHiddenLayer() {
        this.hiddenLayers.push({
            neurons: 4,
            activation: this.config.activation
        });
        this.buildNetwork();
        this.renderHiddenLayers();
        this.resetTraining();
    }

    removeHiddenLayer(index) {
        if (this.hiddenLayers.length > 0) {
            this.hiddenLayers.splice(index, 1);
            this.buildNetwork();
            this.renderHiddenLayers();
            this.resetTraining();
        }
    }

    changeNeuronCount(layerIndex, delta) {
        const layer = this.hiddenLayers[layerIndex];
        const newCount = layer.neurons + delta;

        if (newCount >= 1 && newCount <= 16) {
            layer.neurons = newCount;
            this.buildNetwork();
            this.renderHiddenLayers();
            this.resetTraining();
        }
    }

    updateNetworkSummary() {
        const paramsEl = document.getElementById('total-params');
        const layersEl = document.getElementById('total-layers');

        if (paramsEl && this.network) {
            paramsEl.textContent = this.network.paramCount.toLocaleString();
        }

        if (layersEl) {
            layersEl.textContent = this.hiddenLayers.length + 2; // +2 for input and output
        }
    }

    generateDataset() {
        this.dataset = DatasetGenerator.generate(
            this.config.datasetType,
            this.config.numPoints,
            this.config.noise
        );

        if (this.boundaryViz) {
            this.boundaryViz.setDataset(this.dataset);
        }

        this.updateVisualization();
    }

    buildNetwork() {
        // Build architecture array with dynamic input size
        const inputSize = this.getActiveFeatureCount();
        const architecture = [inputSize];

        for (const layer of this.hiddenLayers) {
            architecture.push(layer.neurons);
        }

        architecture.push(1); // Output layer (1 neuron for binary classification)

        // Create network
        this.network = NeuralNetwork.create(
            architecture,
            this.config.activation,
            'sigmoid'
        );

        // Reset training metrics when network changes
        this.metrics = {
            epoch: 0,
            loss: 0,
            accuracy: 0.5
        };
        this.lossHistory = [];
        this.updateMetricsDisplay();

        // Update visualizations
        if (this.networkViz) {
            this.networkViz.setNetwork(this.network);
            // Pass feature config and transform function to visualizer
            this.networkViz.inputFeatures = this.inputFeatures;
            this.networkViz.transformFeatures = (x1, x2) => this.transformFeatures(x1, x2);
        }

        if (this.boundaryViz) {
            this.boundaryViz.setNetwork(this.network);
            // Pass feature transform function to boundary visualizer
            this.boundaryViz.transformFeatures = (x1, x2) => this.transformFeatures(x1, x2);
        }

        this.updateNetworkSummary();
    }

    toggleTraining() {
        if (this.isTraining) {
            this.stopTraining();
        } else {
            this.startTraining();
        }
    }

    startTraining() {
        this.isTraining = true;
        this.lastUpdateTime = Date.now();
        this.iterationCount = 0;

        // Update button state
        const btnTrain = document.getElementById('btn-train');
        if (btnTrain) {
            btnTrain.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="4" width="4" height="16"/>
                    <rect x="14" y="4" width="4" height="16"/>
                </svg>
                <span>Pause</span>
            `;
            btnTrain.classList.add('btn--training');
        }

        // Training loop
        this.trainingInterval = setInterval(() => {
            this.trainStep();
            this.iterationCount++;

            // Update speed metric every 500ms
            const now = Date.now();
            if (now - this.lastUpdateTime >= 500) {
                this.metrics.speed = Math.round(this.iterationCount / ((now - this.lastUpdateTime) / 1000));
                this.lastUpdateTime = now;
                this.iterationCount = 0;
                this.updateMetricsDisplay();
            }
        }, 16); // ~60fps
    }

    stopTraining() {
        this.isTraining = false;

        if (this.trainingInterval) {
            clearInterval(this.trainingInterval);
            this.trainingInterval = null;
        }

        // Update button state
        const btnTrain = document.getElementById('btn-train');
        if (btnTrain) {
            btnTrain.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
                <span>Train</span>
            `;
            btnTrain.classList.remove('btn--training');
        }
    }

    trainStep() {
        if (!this.network || !this.dataset) return;

        // Get batch (raw x1, x2 coordinates)
        const { X, y } = DatasetGenerator.getBatch(
            this.dataset.X,
            this.dataset.y,
            this.config.batchSize === 'all' ? this.dataset.X.rows : this.config.batchSize
        );

        // Transform to selected features
        const transformedX = this.transformDataset(X);

        // Train step with Adam optimizer
        const result = this.network.trainStepAdam(
            transformedX,
            y,
            this.config.learningRate,
            this.config.regularization
        );

        // Update metrics
        this.metrics.epoch++;
        this.metrics.loss = result.loss;
        this.metrics.accuracy = result.accuracy;

        // Record history
        this.lossHistory.push(result.loss);
        if (this.lossHistory.length > 200) {
            this.lossHistory.shift();
        }

        // Update visualization (throttled)
        if (this.metrics.epoch % 5 === 0) {
            this.updateVisualization();
        }

        this.updateMetricsDisplay();
        this.drawLossChart();
    }

    resetTraining() {
        this.stopTraining();

        if (this.network) {
            this.network.reset();
        }

        this.metrics = {
            epoch: 0,
            loss: 0,
            accuracy: 0.5,
            speed: 0
        };
        this.lossHistory = [];

        this.buildNetwork();
        this.updateVisualization();
        this.updateMetricsDisplay();
        this.drawLossChart();
    }

    updateVisualization() {
        // Update network visualization
        if (this.networkViz && this.network) {
            const activations = this.network.getActivations();
            this.networkViz.updateActivations(activations);
        }

        // Update decision boundary
        if (this.boundaryViz && this.network && this.dataset) {
            // Use imported Tensor class
            this.updateBoundaryWithTensor();
        }
    }

    updateBoundaryWithTensor() {
        const { min, max } = { min: -1.2, max: 1.2 };
        const resolution = 50;
        const step = (max - min) / resolution;
        const numFeatures = this.getActiveFeatureCount();

        // Create grid of transformed features
        const transformedPoints = new Float32Array(resolution * resolution * numFeatures);

        let idx = 0;
        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const x1 = min + j * step + step / 2;
                const x2 = max - i * step - step / 2;
                const features = this.transformFeatures(x1, x2);

                for (const f of features) {
                    transformedPoints[idx++] = f;
                }
            }
        }

        const inputTensor = new Tensor(resolution * resolution, numFeatures, transformedPoints);
        const predictions = this.network.predict(inputTensor);

        this.boundaryViz.heatmapData = Array.from(predictions.data);
        this.boundaryViz.renderHeatmapToImage();
        this.boundaryViz.render();
    }

    updateMetricsDisplay() {
        const epochEl = document.getElementById('metric-epoch');
        const lossEl = document.getElementById('metric-loss');
        const accuracyEl = document.getElementById('metric-accuracy');
        const speedEl = document.getElementById('metric-speed');

        if (epochEl) epochEl.textContent = this.metrics.epoch.toLocaleString();
        if (lossEl) lossEl.textContent = this.metrics.loss.toFixed(4);
        if (accuracyEl) accuracyEl.textContent = (this.metrics.accuracy * 100).toFixed(1) + '%';
        if (speedEl) speedEl.textContent = `${this.metrics.speed} iter/s`;
    }

    drawLossChart() {
        if (!this.lossCanvas || !this.lossCtx) return;

        const ctx = this.lossCtx;
        const dpr = window.devicePixelRatio || 1;
        const rect = this.lossCanvas.getBoundingClientRect();

        this.lossCanvas.width = rect.width * dpr;
        this.lossCanvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const width = rect.width;
        const height = rect.height;
        const padding = 5;

        // Clear
        ctx.clearRect(0, 0, width, height);

        if (this.lossHistory.length < 2) return;

        // Find min/max for scaling
        const maxLoss = Math.max(...this.lossHistory, 0.1);
        const minLoss = Math.min(...this.lossHistory, 0);

        // Draw line
        ctx.beginPath();
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;

        for (let i = 0; i < this.lossHistory.length; i++) {
            const x = padding + (i / (this.lossHistory.length - 1)) * (width - padding * 2);
            const y = padding + (1 - (this.lossHistory[i] - minLoss) / (maxLoss - minLoss)) * (height - padding * 2);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();

        // Draw gradient fill under line
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, 'rgba(245, 158, 11, 0.25)');
        gradient.addColorStop(1, 'rgba(245, 158, 11, 0)');

        ctx.lineTo(width - padding, height - padding);
        ctx.lineTo(padding, height - padding);
        ctx.closePath();
        ctx.fillStyle = gradient;
        ctx.fill();
    }
}

// Modal Controller for Tutorial and Examples
class ModalController {
    constructor() {
        this.tutorialModal = document.getElementById('tutorial-modal');
        this.examplesModal = document.getElementById('examples-modal');
        this.init();
    }

    init() {
        // Tutorial button
        const btnTutorial = document.getElementById('btn-tutorial');
        if (btnTutorial) {
            btnTutorial.addEventListener('click', () => this.openTutorial());
        }

        // Examples button
        const btnExamples = document.getElementById('btn-examples');
        if (btnExamples) {
            btnExamples.addEventListener('click', () => this.openExamples());
        }

        // Close buttons
        document.getElementById('tutorial-close-btn')?.addEventListener('click', () => this.closeTutorial());
        document.getElementById('examples-close-btn')?.addEventListener('click', () => this.closeExamples());

        // Close on backdrop click
        this.tutorialModal?.addEventListener('click', (e) => {
            if (e.target === this.tutorialModal) this.closeTutorial();
        });
        this.examplesModal?.addEventListener('click', (e) => {
            if (e.target === this.examplesModal) this.closeExamples();
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (this.tutorialModal?.classList.contains('active')) this.closeTutorial();
                if (this.examplesModal?.classList.contains('active')) this.closeExamples();
            }
        });

        // Setup example loading function
        window.loadExample = (example) => this.loadExample(example);
    }

    openTutorial() {
        this.tutorialModal?.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    closeTutorial() {
        this.tutorialModal?.classList.remove('active');
        document.body.style.overflow = '';
    }

    openExamples() {
        this.examplesModal?.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    closeExamples() {
        this.examplesModal?.classList.remove('active');
        document.body.style.overflow = '';
    }

    loadExample(example) {
        this.closeExamples();

        switch (example) {
            case 'xor':
                // Switch to MLP tab and load XOR dataset
                window.tabController.switchTab('mlp');
                setTimeout(() => {
                    const xorBtn = document.querySelector('[data-dataset="xor"]');
                    if (xorBtn) xorBtn.click();
                }, 100);
                break;

            case 'spiral':
                // Switch to MLP tab and load Spiral dataset
                window.tabController.switchTab('mlp');
                setTimeout(() => {
                    const spiralBtn = document.querySelector('[data-dataset="spiral"]');
                    if (spiralBtn) spiralBtn.click();
                }, 100);
                break;

            case 'mnist':
                // Switch to CNN tab
                window.tabController.switchTab('cnn');
                break;

            case 'transformer':
                // Switch to Transformer tab
                window.tabController.switchTab('transformer');
                break;

            case 'diffusion':
                // Switch to Diffusion tab
                window.tabController.switchTab('diffusion');
                break;
        }
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.playground = new NeuralPlayground();
    window.tabController = new TabController();
    window.modalController = new ModalController();
});
