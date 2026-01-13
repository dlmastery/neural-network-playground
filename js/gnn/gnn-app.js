/**
 * GNN App Controller
 * Main controller for the Graph Neural Network tab
 */

import { loadDataset } from './gnn-dataset.js';
import { GNNModel, GNN_LAYER_TYPES } from './gnn-model.js';
import { GNNVisualizer } from './gnn-viz.js';

export class GNNApp {
    constructor() {
        // Data
        this.dataset = null;
        this.model = null;
        this.viz = null;
        this.currentState = null;

        // UI state
        this.selectedNode = null;
        this.currentLayer = 0;
        this.isAnimating = false;
        this.isTraining = false;
        this.trainingFrame = null;

        // Settings
        this.layerType = 'gcn';
        this.numLayers = 2;
        this.hiddenDim = 16;
        this.learningRate = 0.01;

        // Training stats
        this.epoch = 0;
        this.trainHistory = { loss: [], accuracy: [] };

        // DOM elements
        this.elements = {};
    }

    /**
     * Initialize the GNN app
     */
    async init() {
        console.log('Initializing GNN App...');

        this.cacheElements();
        this.setupEventListeners();

        // Load default dataset
        this.loadDataset('karate');

        // Build model
        this.buildModel();

        // Initialize visualization
        this.initVisualization();

        // Run initial forward pass
        this.runForward();

        console.log('GNN App initialized');
    }

    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements = {
            // Controls
            datasetSelect: document.getElementById('gnn-dataset-select'),
            archSelect: document.getElementById('gnn-arch-select'),
            layersSlider: document.getElementById('gnn-layers'),
            layersValue: document.getElementById('gnn-layers-value'),
            trainBtn: document.getElementById('gnn-train-btn'),
            stopBtn: document.getElementById('gnn-stop-btn'),

            // Animation controls
            playBtn: document.getElementById('gnn-play-btn'),
            stepBtn: document.getElementById('gnn-step-btn'),
            resetBtn: document.getElementById('gnn-reset-btn'),
            layerProgress: document.getElementById('gnn-layer-progress'),
            currentLayerDisplay: document.getElementById('gnn-current-layer'),

            // Canvases
            graphCanvas: document.getElementById('gnn-graph-canvas'),
            messageCanvas: document.getElementById('gnn-message-canvas'),
            embeddingCanvas: document.getElementById('gnn-embedding-canvas'),

            // Info displays
            nodeCount: document.getElementById('gnn-node-count'),
            edgeCount: document.getElementById('gnn-edge-count'),
            epochDisplay: document.getElementById('gnn-epoch'),
            lossDisplay: document.getElementById('gnn-loss'),
            accuracyDisplay: document.getElementById('gnn-accuracy'),
            nodeDetails: document.getElementById('gnn-node-details')
        };
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Dataset selection
        this.elements.datasetSelect?.addEventListener('change', (e) => {
            this.loadDataset(e.target.value);
            this.buildModel();
            this.runForward();
        });

        // Architecture selection
        this.elements.archSelect?.addEventListener('change', (e) => {
            this.layerType = e.target.value;
            this.buildModel();
            this.runForward();
        });

        // Layers slider
        this.elements.layersSlider?.addEventListener('input', (e) => {
            this.numLayers = parseInt(e.target.value);
            if (this.elements.layersValue) {
                this.elements.layersValue.textContent = this.numLayers;
            }
            this.buildModel();
            this.runForward();
        });

        // Train button
        this.elements.trainBtn?.addEventListener('click', () => {
            if (this.isTraining) {
                this.stopTraining();
            } else {
                this.startTraining();
            }
        });

        // Stop button
        this.elements.stopBtn?.addEventListener('click', () => {
            this.stopTraining();
        });

        // Animation: Play
        this.elements.playBtn?.addEventListener('click', () => {
            this.playAnimation();
        });

        // Animation: Step
        this.elements.stepBtn?.addEventListener('click', () => {
            this.stepLayer();
        });

        // Animation: Reset
        this.elements.resetBtn?.addEventListener('click', () => {
            this.resetAnimation();
        });

        // Graph canvas click - node selection
        this.elements.graphCanvas?.addEventListener('click', (e) => {
            this.handleGraphClick(e);
        });

        // Graph canvas hover
        this.elements.graphCanvas?.addEventListener('mousemove', (e) => {
            this.handleGraphHover(e);
        });
    }

    /**
     * Load dataset
     */
    loadDataset(name) {
        try {
            this.dataset = loadDataset(name);

            // Update info displays
            if (this.elements.nodeCount) {
                this.elements.nodeCount.textContent = this.dataset.numNodes;
            }
            if (this.elements.edgeCount) {
                this.elements.edgeCount.textContent = this.dataset.numEdges;
            }

            // Reset training
            this.epoch = 0;
            this.trainHistory = { loss: [], accuracy: [] };
            this.updateMetricsDisplay();

            console.log(`Loaded dataset: ${name} (${this.dataset.numNodes} nodes, ${this.dataset.numEdges} edges)`);
        } catch (error) {
            console.error('Failed to load dataset:', error);
        }
    }

    /**
     * Build GNN model
     */
    buildModel() {
        if (!this.dataset) return;

        this.model = new GNNModel({
            inputDim: this.dataset.featureDim,
            hiddenDim: this.hiddenDim,
            outputDim: this.dataset.numClasses,
            numLayers: this.numLayers,
            layerType: this.layerType
        });

        console.log(`Built ${this.layerType.toUpperCase()} model with ${this.numLayers} layers`);
    }

    /**
     * Initialize visualization
     */
    initVisualization() {
        this.viz = new GNNVisualizer(
            this.elements.graphCanvas,
            this.elements.messageCanvas,
            this.elements.embeddingCanvas
        );

        if (this.dataset) {
            this.viz.setGraph(
                this.dataset.numNodes,
                this.dataset.edges,
                this.dataset.neighbors,
                this.dataset.labels
            );

            // Run layout to stability
            this.viz.runLayoutToStable(200);
        }
    }

    /**
     * Run forward pass and update visualization
     */
    runForward() {
        if (!this.model || !this.dataset) return;

        this.currentState = this.model.forward(
            this.dataset.nodeFeatures,
            this.dataset.adjNormalized,
            this.dataset.adjMatrix,
            this.dataset.numNodes
        );

        // Update visualizations
        this.viz.setPredictions(this.currentState.predictions);

        // Set attention weights if GAT
        if (this.layerType === 'gat' && this.currentState.layers.length > 0) {
            const lastLayer = this.currentState.layers[this.currentLayer];
            this.viz.setAttentionWeights(lastLayer?.attentionWeights || null);
        } else {
            this.viz.setAttentionWeights(null);
        }

        this.updateVisualizations();
    }

    /**
     * Update all visualizations
     */
    updateVisualizations() {
        if (!this.viz || !this.currentState) return;

        // Render graph
        this.viz.renderGraph();

        // Render message passing for current layer
        const layerState = this.currentState.layers[this.currentLayer];
        this.viz.renderMessagePassing(layerState, 1);

        // Render embeddings
        const embeddings = layerState?.output || this.dataset.nodeFeatures;
        const dim = this.currentLayer === this.numLayers - 1 ?
            this.dataset.numClasses :
            (this.currentLayer === 0 && this.numLayers === 1 ? this.dataset.numClasses : this.hiddenDim);

        this.viz.renderEmbeddings(
            embeddings,
            this.dataset.numNodes,
            dim
        );
    }

    /**
     * Start training
     */
    startTraining() {
        if (this.isTraining) return;

        this.isTraining = true;

        if (this.elements.trainBtn) {
            this.elements.trainBtn.textContent = 'Pause';
        }
        if (this.elements.stopBtn) {
            this.elements.stopBtn.disabled = false;
        }

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

        if (this.elements.trainBtn) {
            this.elements.trainBtn.textContent = 'Train';
        }
        if (this.elements.stopBtn) {
            this.elements.stopBtn.disabled = true;
        }
    }

    /**
     * Training loop
     */
    trainingLoop() {
        if (!this.isTraining) return;

        // Train step
        const result = this.model.trainStep(
            this.dataset.nodeFeatures,
            this.dataset.adjNormalized,
            this.dataset.adjMatrix,
            this.dataset.labels,
            this.dataset.numNodes,
            this.learningRate
        );

        this.epoch++;
        this.trainHistory.loss.push(result.loss);
        this.trainHistory.accuracy.push(result.accuracy);

        // Update displays
        this.updateMetricsDisplay();

        // Run forward pass to update state
        this.runForward();

        // Continue training
        this.trainingFrame = requestAnimationFrame(() => this.trainingLoop());
    }

    /**
     * Update metrics display
     */
    updateMetricsDisplay() {
        if (this.elements.epochDisplay) {
            this.elements.epochDisplay.textContent = this.epoch;
        }
        if (this.elements.lossDisplay) {
            const loss = this.trainHistory.loss.length > 0 ?
                this.trainHistory.loss[this.trainHistory.loss.length - 1] : '-';
            this.elements.lossDisplay.textContent = typeof loss === 'number' ?
                loss.toFixed(4) : loss;
        }
        if (this.elements.accuracyDisplay) {
            const acc = this.trainHistory.accuracy.length > 0 ?
                this.trainHistory.accuracy[this.trainHistory.accuracy.length - 1] : '-';
            this.elements.accuracyDisplay.textContent = typeof acc === 'number' ?
                (acc * 100).toFixed(1) + '%' : acc;
        }
    }

    /**
     * Step to next layer
     */
    stepLayer() {
        this.currentLayer = (this.currentLayer + 1) % this.numLayers;
        this.viz.setCurrentLayer(this.currentLayer);

        // Update attention weights for GAT
        if (this.layerType === 'gat' && this.currentState?.layers) {
            const layerState = this.currentState.layers[this.currentLayer];
            this.viz.setAttentionWeights(layerState?.attentionWeights || null);
        }

        this.updateLayerDisplay();
        this.updateVisualizations();
    }

    /**
     * Play animation through all layers
     */
    playAnimation() {
        if (this.isAnimating) {
            this.viz.stopAnimation();
            this.isAnimating = false;
            if (this.elements.playBtn) {
                this.elements.playBtn.textContent = '\u25B6';
            }
            return;
        }

        this.isAnimating = true;
        if (this.elements.playBtn) {
            this.elements.playBtn.textContent = '\u23F8';
        }

        this.currentLayer = 0;
        this.animateLayer();
    }

    /**
     * Animate current layer
     */
    animateLayer() {
        if (!this.isAnimating) return;

        this.viz.setCurrentLayer(this.currentLayer);

        // Update attention weights for GAT
        if (this.layerType === 'gat' && this.currentState?.layers) {
            const layerState = this.currentState.layers[this.currentLayer];
            this.viz.setAttentionWeights(layerState?.attentionWeights || null);
        }

        this.updateLayerDisplay();

        // Animate message passing
        this.viz.startAnimation((progress) => {
            const layerState = this.currentState?.layers[this.currentLayer];
            this.viz.renderMessagePassing(layerState, progress);
            this.updateProgressBar(progress);
        });

        // After animation completes, move to next layer
        setTimeout(() => {
            if (!this.isAnimating) return;

            this.currentLayer++;

            if (this.currentLayer >= this.numLayers) {
                // Animation complete
                this.isAnimating = false;
                this.currentLayer = this.numLayers - 1;
                if (this.elements.playBtn) {
                    this.elements.playBtn.textContent = '\u25B6';
                }
                this.updateVisualizations();
            } else {
                // Continue to next layer
                this.animateLayer();
            }
        }, 1500);
    }

    /**
     * Reset animation
     */
    resetAnimation() {
        this.viz.stopAnimation();
        this.isAnimating = false;
        this.currentLayer = 0;
        this.viz.setCurrentLayer(0);

        if (this.elements.playBtn) {
            this.elements.playBtn.textContent = '\u25B6';
        }

        // Reset attention weights
        if (this.layerType === 'gat' && this.currentState?.layers) {
            const layerState = this.currentState.layers[0];
            this.viz.setAttentionWeights(layerState?.attentionWeights || null);
        }

        this.updateLayerDisplay();
        this.updateProgressBar(0);
        this.updateVisualizations();
    }

    /**
     * Update layer display
     */
    updateLayerDisplay() {
        if (this.elements.currentLayerDisplay) {
            this.elements.currentLayerDisplay.textContent = this.currentLayer + 1;
        }
    }

    /**
     * Update progress bar
     */
    updateProgressBar(progress) {
        if (this.elements.layerProgress) {
            const totalProgress = (this.currentLayer + progress) / this.numLayers * 100;
            this.elements.layerProgress.style.width = `${totalProgress}%`;
        }
    }

    /**
     * Handle graph canvas click
     */
    handleGraphClick(event) {
        const rect = this.elements.graphCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const nodeId = this.viz.getNodeAtPosition(x, y);
        this.selectNode(nodeId);
    }

    /**
     * Handle graph canvas hover
     */
    handleGraphHover(event) {
        const rect = this.elements.graphCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const nodeId = this.viz.getNodeAtPosition(x, y);

        if (nodeId !== null) {
            this.elements.graphCanvas.style.cursor = 'pointer';
        } else {
            this.elements.graphCanvas.style.cursor = 'default';
        }
    }

    /**
     * Select a node
     */
    selectNode(nodeId) {
        this.selectedNode = nodeId;
        this.viz.setSelectedNode(nodeId);
        this.updateNodeDetails();
        this.updateVisualizations();
    }

    /**
     * Create a detail row element safely
     */
    createDetailRow(label, value, valueClass = '') {
        const row = document.createElement('div');
        row.className = 'node-detail-row';

        const labelSpan = document.createElement('span');
        labelSpan.className = 'label';
        labelSpan.textContent = label;

        const valueSpan = document.createElement('span');
        valueSpan.className = 'value' + (valueClass ? ' ' + valueClass : '');
        valueSpan.textContent = value;

        row.appendChild(labelSpan);
        row.appendChild(valueSpan);

        return row;
    }

    /**
     * Update node details panel using safe DOM methods
     */
    updateNodeDetails() {
        if (!this.elements.nodeDetails) return;

        // Clear previous content safely
        while (this.elements.nodeDetails.firstChild) {
            this.elements.nodeDetails.removeChild(this.elements.nodeDetails.firstChild);
        }

        if (this.selectedNode === null) {
            const hint = document.createElement('p');
            hint.className = 'gnn-node-hint';
            hint.textContent = 'Click a node to inspect';
            this.elements.nodeDetails.appendChild(hint);
            return;
        }

        const nodeId = this.selectedNode;
        const label = this.dataset.labels[nodeId];
        const predicted = this.currentState?.predictions[nodeId];
        const neighbors = this.dataset.getNeighbors(nodeId);

        // Node ID
        this.elements.nodeDetails.appendChild(
            this.createDetailRow('Node ID:', nodeId.toString())
        );

        // True label
        this.elements.nodeDetails.appendChild(
            this.createDetailRow('True Class:', this.dataset.classNames[label])
        );

        // Prediction
        if (predicted !== undefined) {
            const isCorrect = predicted === label;
            this.elements.nodeDetails.appendChild(
                this.createDetailRow(
                    'Predicted:',
                    this.dataset.classNames[predicted],
                    isCorrect ? 'correct' : 'incorrect'
                )
            );
        }

        // Degree
        this.elements.nodeDetails.appendChild(
            this.createDetailRow('Degree:', neighbors.length.toString())
        );

        // Neighbors
        const neighborsText = neighbors.slice(0, 8).join(', ') + (neighbors.length > 8 ? '...' : '');
        this.elements.nodeDetails.appendChild(
            this.createDetailRow('Neighbors:', neighborsText)
        );
    }

    /**
     * Called when tab becomes active
     */
    onActivate() {
        // Update layout if needed
        if (this.viz && !this.viz.isLayoutStable) {
            this.viz.runLayoutToStable(50);
        }
        this.updateVisualizations();
    }

    /**
     * Dispose resources
     */
    dispose() {
        this.stopTraining();
        this.viz?.dispose();
    }
}

// Export for main.js
export async function initGNNApp() {
    const app = new GNNApp();
    await app.init();
    return app;
}
