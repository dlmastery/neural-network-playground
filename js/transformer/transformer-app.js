/**
 * Transformer App Controller
 * Main controller for the Transformer Playground tab
 */

import { TransformerModel } from './transformer-model.js';
import { TransformerViz } from './transformer-viz.js';
import { TransformerExplainer } from './transformer-explainer.js';

export class TransformerApp {
    constructor() {
        this.model = new TransformerModel();
        this.viz = null;
        this.explainer = null;

        // UI elements
        this.elements = {};

        // State
        this.isGenerating = false;
        this.abortController = null;
        this.generatedTokens = [];

        // Settings
        this.settings = {
            modelId: 'Xenova/distilgpt2',
            device: 'webgpu',  // GPU by default, falls back to WASM if unavailable
            temperature: 0.7,
            maxLength: 50
        };
    }

    /**
     * Initialize the transformer app
     */
    async init() {
        console.log('Initializing Transformer Playground...');

        // Cache UI elements
        this.cacheElements();

        // Initialize visualization
        this.initVisualization();

        // Initialize explainer
        this.initExplainer();

        // Setup event listeners
        this.setupEventListeners();

        // Update UI with initial model info
        this.updateModelInfo();

        // Check WebGPU availability
        await this.checkWebGPU();

        // Don't auto-load model - wait for user to switch to tab
        console.log('Transformer Playground ready (model will load on first use)');
    }

    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements = {
            // Controls
            modelSelect: document.getElementById('transformer-model-select'),
            deviceSelect: document.getElementById('transformer-device-select'),
            deviceStatus: document.getElementById('transformer-device-status'),
            generateBtn: document.getElementById('transformer-generate-btn'),
            stopBtn: document.getElementById('transformer-stop-btn'),
            explainBtn: document.getElementById('transformer-explain-btn'),
            temperatureSlider: document.getElementById('transformer-temperature'),
            tempValue: document.getElementById('transformer-temp-value'),
            maxLengthInput: document.getElementById('transformer-max-length'),

            // Status
            statusDisplay: document.getElementById('transformer-status'),
            tokensDisplay: document.getElementById('transformer-tokens'),

            // Loading
            loadingProgress: document.getElementById('transformer-loading-progress'),
            loadingStatus: document.getElementById('transformer-loading-status'),
            loadingBar: document.getElementById('transformer-loading-bar'),

            // Model info
            modelName: document.getElementById('model-name'),
            modelParams: document.getElementById('model-params'),
            modelLayers: document.getElementById('model-layers'),
            modelHidden: document.getElementById('model-hidden'),
            modelHeads: document.getElementById('model-heads'),
            modelVocab: document.getElementById('model-vocab'),

            // Input/Output
            inputTextarea: document.getElementById('transformer-input'),
            outputDiv: document.getElementById('transformer-output'),

            // Canvases
            archCanvas: document.getElementById('transformer-arch-canvas'),
            attentionCanvas: document.getElementById('transformer-attention-canvas'),

            // Attention controls
            attentionLayerSelect: document.getElementById('transformer-attention-layer'),
            attentionHeadSelect: document.getElementById('transformer-attention-head'),
            attentionTokens: document.getElementById('transformer-attention-tokens'),

            // Explainer modal
            explainerModal: document.getElementById('transformer-explainer-modal'),
            explainerCloseBtn: document.getElementById('transformer-explainer-close-btn')
        };
    }

    /**
     * Initialize visualization components
     */
    initVisualization() {
        this.viz = new TransformerViz(
            this.elements.archCanvas,
            this.elements.attentionCanvas
        );
        this.viz.init();
    }

    /**
     * Initialize explainer modal
     */
    initExplainer() {
        this.explainer = new TransformerExplainer(this.model);
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Generate button
        this.elements.generateBtn?.addEventListener('click', () => this.generate());

        // Stop button
        this.elements.stopBtn?.addEventListener('click', () => this.stopGeneration());

        // Explain button
        this.elements.explainBtn?.addEventListener('click', () => this.openExplainer());

        // Model selection
        this.elements.modelSelect?.addEventListener('change', (e) => {
            this.settings.modelId = e.target.value;
            this.updateModelInfo();
            // Model will be reloaded on next generate
            this.model.isLoaded = false;
        });

        // Device selection
        this.elements.deviceSelect?.addEventListener('change', (e) => {
            this.settings.device = e.target.value;
            // Model will be reloaded on next generate
            this.model.isLoaded = false;
        });

        // Temperature slider
        this.elements.temperatureSlider?.addEventListener('input', (e) => {
            this.settings.temperature = parseFloat(e.target.value);
            if (this.elements.tempValue) {
                this.elements.tempValue.textContent = this.settings.temperature.toFixed(1);
            }
        });

        // Max length input
        this.elements.maxLengthInput?.addEventListener('change', (e) => {
            this.settings.maxLength = parseInt(e.target.value);
        });

        // Attention layer/head selection
        this.elements.attentionLayerSelect?.addEventListener('change', (e) => {
            this.viz.setSelection(parseInt(e.target.value), this.viz.selectedHead);
        });

        this.elements.attentionHeadSelect?.addEventListener('change', (e) => {
            this.viz.setSelection(this.viz.selectedLayer, parseInt(e.target.value));
        });

        // Explainer close button
        this.elements.explainerCloseBtn?.addEventListener('click', () => this.closeExplainer());

        // Close modal on backdrop click
        this.elements.explainerModal?.addEventListener('click', (e) => {
            if (e.target === this.elements.explainerModal) {
                this.closeExplainer();
            }
        });

        // Model progress callback
        this.model.onProgress = (progress) => this.updateLoadingProgress(progress);
        this.model.onStatusChange = (status) => this.updateStatus(status);
    }

    /**
     * Check WebGPU availability and update UI
     */
    async checkWebGPU() {
        const hasWebGPU = await this.model.checkWebGPU();

        if (this.elements.deviceSelect) {
            const webgpuOption = this.elements.deviceSelect.querySelector('option[value="webgpu"]');
            if (webgpuOption) {
                if (hasWebGPU) {
                    webgpuOption.textContent = 'WebGPU (GPU)';
                } else {
                    webgpuOption.textContent = 'WebGPU (Not Available)';
                    webgpuOption.disabled = true;
                }
            }
        }

        if (this.elements.deviceStatus) {
            this.elements.deviceStatus.textContent = hasWebGPU ? '✓' : '';
            this.elements.deviceStatus.title = hasWebGPU
                ? 'WebGPU available'
                : 'WebGPU not available';
        }
    }

    /**
     * Update model info display
     */
    updateModelInfo() {
        const info = this.model.modelInfo[this.settings.modelId];
        if (!info) return;

        if (this.elements.modelName) this.elements.modelName.textContent = info.name;
        if (this.elements.modelParams) this.elements.modelParams.textContent = info.params;
        if (this.elements.modelLayers) this.elements.modelLayers.textContent = info.layers;
        if (this.elements.modelHidden) this.elements.modelHidden.textContent = info.hidden;
        if (this.elements.modelHeads) this.elements.modelHeads.textContent = info.heads;
        if (this.elements.modelVocab) this.elements.modelVocab.textContent = info.vocab.toLocaleString();

        // Update visualization
        if (this.viz) {
            this.viz.setModelConfig({
                layers: info.layers,
                heads: info.heads,
                hidden: info.hidden
            });
        }
    }

    /**
     * Update loading progress
     */
    updateLoadingProgress(progress) {
        if (this.elements.loadingProgress) {
            this.elements.loadingProgress.style.display = 'block';
        }
        if (this.elements.loadingStatus) {
            this.elements.loadingStatus.textContent = progress.status;
        }
        if (this.elements.loadingBar) {
            this.elements.loadingBar.style.width = `${progress.percentage}%`;
        }

        if (progress.percentage >= 100) {
            setTimeout(() => {
                if (this.elements.loadingProgress) {
                    this.elements.loadingProgress.style.display = 'none';
                }
            }, 500);
        }
    }

    /**
     * Update status display
     */
    updateStatus(status) {
        if (this.elements.statusDisplay) {
            this.elements.statusDisplay.textContent = status;
        }
    }

    /**
     * Load model if not loaded
     */
    async ensureModelLoaded() {
        if (!this.model.isLoaded && !this.model.isLoading) {
            await this.model.init(this.settings.modelId, this.settings.device);
        }
    }

    /**
     * Generate text
     */
    async generate() {
        if (this.isGenerating) return;

        const prompt = this.elements.inputTextarea?.value?.trim();
        if (!prompt) {
            this.updateStatus('Please enter a prompt');
            return;
        }

        this.isGenerating = true;
        this.generatedTokens = [];

        // Update UI
        this.elements.generateBtn.disabled = true;
        this.elements.stopBtn.disabled = false;

        // Clear output using safe DOM methods
        this.clearOutput();

        try {
            // Load model if needed
            await this.ensureModelLoaded();

            // Tokenize prompt for visualization
            const { tokens } = this.model.tokenize(prompt);
            this.viz.setTokens(tokens);

            // Generate mock attention for visualization (real attention would require model access)
            const mockAttention = this.viz.generateMockAttention(tokens.length);
            this.viz.setAttentionWeights(mockAttention);

            // Start data flow animation
            this.viz.animateDataFlow();

            // Display prompt tokens
            this.displayTokens(prompt, tokens, []);

            // Generate text
            const result = await this.model.generateSync(prompt, {
                maxNewTokens: this.settings.maxLength,
                temperature: this.settings.temperature,
                topK: 50,
                topP: 0.95,
                doSample: this.settings.temperature > 0
            });

            // Display result
            const { tokens: allTokens } = this.model.tokenize(result);
            const promptTokens = tokens;
            const generatedTokens = allTokens.slice(promptTokens.length);

            this.displayTokens(prompt, promptTokens, generatedTokens);

            // Update token count
            if (this.elements.tokensDisplay) {
                this.elements.tokensDisplay.textContent = allTokens.length;
            }

            // Update attention with full sequence
            this.viz.setTokens(allTokens);
            const fullAttention = this.viz.generateMockAttention(allTokens.length);
            this.viz.setAttentionWeights(fullAttention);

            // Update attention tokens display
            this.updateAttentionTokens(allTokens);

        } catch (error) {
            console.error('Generation error:', error);
            this.updateStatus('Error: ' + error.message);
        } finally {
            this.isGenerating = false;
            this.elements.generateBtn.disabled = false;
            this.elements.stopBtn.disabled = true;
            this.viz.stopAnimation();
        }
    }

    /**
     * Clear output div safely
     */
    clearOutput() {
        if (this.elements.outputDiv) {
            while (this.elements.outputDiv.firstChild) {
                this.elements.outputDiv.removeChild(this.elements.outputDiv.firstChild);
            }
        }
    }

    /**
     * Display tokens in output using safe DOM methods
     */
    displayTokens(prompt, promptTokens, generatedTokens) {
        if (!this.elements.outputDiv) return;

        this.clearOutput();

        // Prompt tokens
        for (const token of promptTokens) {
            const span = document.createElement('span');
            span.className = 'token token--prompt';
            span.textContent = token;
            span.title = `Prompt token: "${token}"`;
            this.elements.outputDiv.appendChild(span);
        }

        // Generated tokens
        for (const token of generatedTokens) {
            const span = document.createElement('span');
            span.className = 'token token--generated';
            span.textContent = token;
            span.title = `Generated token: "${token}"`;
            this.elements.outputDiv.appendChild(span);
        }
    }

    /**
     * Update attention tokens display using safe DOM methods
     */
    updateAttentionTokens(tokens) {
        if (!this.elements.attentionTokens) return;

        // Clear safely
        while (this.elements.attentionTokens.firstChild) {
            this.elements.attentionTokens.removeChild(this.elements.attentionTokens.firstChild);
        }

        const maxDisplay = 10;
        const displayTokens = tokens.slice(0, maxDisplay);

        for (const token of displayTokens) {
            const span = document.createElement('span');
            span.className = 'attention-token';
            span.textContent = token.trim() || '␣';
            this.elements.attentionTokens.appendChild(span);
        }

        if (tokens.length > maxDisplay) {
            const span = document.createElement('span');
            span.className = 'attention-token-more';
            span.textContent = `+${tokens.length - maxDisplay} more`;
            this.elements.attentionTokens.appendChild(span);
        }
    }

    /**
     * Stop generation
     */
    stopGeneration() {
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
        this.isGenerating = false;
        this.elements.generateBtn.disabled = false;
        this.elements.stopBtn.disabled = true;
        this.updateStatus('Stopped');
        this.viz.stopAnimation();
    }

    /**
     * Open explainer modal
     */
    openExplainer() {
        if (this.elements.explainerModal) {
            this.elements.explainerModal.classList.add('active');
            document.body.style.overflow = 'hidden';
            this.explainer?.init();
        }
    }

    /**
     * Close explainer modal
     */
    closeExplainer() {
        if (this.elements.explainerModal) {
            this.elements.explainerModal.classList.remove('active');
            document.body.style.overflow = '';
        }
    }

    /**
     * Called when tab becomes active
     */
    onActivate() {
        // Resize canvases
        if (this.viz) {
            this.viz.resizeCanvases();
            this.viz.renderArchitecture();
        }
    }

    /**
     * Dispose resources
     */
    dispose() {
        this.model.dispose();
        this.viz?.dispose();
    }
}

// Export singleton
export const transformerApp = new TransformerApp();
