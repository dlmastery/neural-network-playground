/**
 * Transformer Explainer Full
 * Interactive full-page controller for the Transformer Explainer
 * Orchestrates PicoTransformer and visualizations
 */

import { PicoTransformer, DEMO_VOCAB } from './pico-transformer.js';
import { ExplainerViz } from './explainer-viz.js';

export class TransformerExplainerFull {
    constructor() {
        // PicoTransformer instance
        this.pico = new PicoTransformer({
            vocabSize: DEMO_VOCAB.length,
            contextLength: 64,
            embeddingDim: 64,
            numHeads: 4,
            numLayers: 2,
            ffnDim: 256
        });

        // Visualization handler
        this.viz = new ExplainerViz();

        // State
        this.currentText = 'The quick brown fox';
        this.promptTokens = [];
        this.promptTokenIds = [];
        this.generatedTokens = [];
        this.generatedTokenIds = [];
        this.currentState = null;

        // Settings
        this.temperature = 1.0;
        this.topK = 5;
        this.selectedLayer = 0;

        // Generation state
        this.isGenerating = false;
        this.generationInterval = null;

        // DOM elements cache
        this.elements = {};

        // Initialized flag
        this.isInitialized = false;
    }

    /**
     * Initialize the explainer
     */
    init() {
        if (this.isInitialized) return;

        this.cacheElements();
        this.setupEventListeners();
        this.processInput();

        this.isInitialized = true;
        console.log('Transformer Explainer Full initialized');
    }

    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements = {
            // Main container
            container: document.getElementById('transformer-explainer-view'),

            // Input
            inputText: document.getElementById('explainer-input-text'),

            // Canvases
            embeddingsCanvas: document.getElementById('explainer-embeddings-canvas'),
            attentionCanvas: document.getElementById('explainer-attention-canvas'),
            ffnCanvas: document.getElementById('explainer-ffn-canvas'),
            probabilitiesCanvas: document.getElementById('explainer-probabilities-canvas'),
            tokensCanvas: document.getElementById('explainer-tokens-canvas'),

            // Controls
            temperatureSlider: document.getElementById('explainer-temperature'),
            temperatureValue: document.getElementById('explainer-temp-value'),
            layerSelect: document.getElementById('explainer-layer-select'),
            stepBtn: document.getElementById('explainer-step-btn'),
            autoBtn: document.getElementById('explainer-auto-btn'),
            resetBtn: document.getElementById('explainer-reset-btn'),
            backBtn: document.getElementById('explainer-back-btn'),

            // Info displays
            tokenCount: document.getElementById('explainer-token-count'),
            currentToken: document.getElementById('explainer-current-token')
        };
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Input text change
        if (this.elements.inputText) {
            this.elements.inputText.addEventListener('input', (e) => {
                this.currentText = e.target.value;
                this.resetGeneration();
                this.processInput();
            });
        }

        // Temperature slider
        if (this.elements.temperatureSlider) {
            this.elements.temperatureSlider.addEventListener('input', (e) => {
                this.temperature = parseFloat(e.target.value);
                if (this.elements.temperatureValue) {
                    this.elements.temperatureValue.textContent = this.temperature.toFixed(1);
                }
                // Re-run with new temperature
                this.updatePredictions();
            });
        }

        // Layer selection
        if (this.elements.layerSelect) {
            this.elements.layerSelect.addEventListener('change', (e) => {
                this.selectedLayer = parseInt(e.target.value);
                this.updateVisualizations();
            });
        }

        // Step button
        if (this.elements.stepBtn) {
            this.elements.stepBtn.addEventListener('click', () => {
                this.generateStep();
            });
        }

        // Auto button
        if (this.elements.autoBtn) {
            this.elements.autoBtn.addEventListener('click', () => {
                this.toggleAutoGenerate();
            });
        }

        // Reset button
        if (this.elements.resetBtn) {
            this.elements.resetBtn.addEventListener('click', () => {
                this.resetGeneration();
                this.processInput();
            });
        }

        // Back button
        if (this.elements.backBtn) {
            this.elements.backBtn.addEventListener('click', () => {
                this.close();
            });
        }

        // Attention canvas hover
        if (this.elements.attentionCanvas) {
            this.elements.attentionCanvas.addEventListener('mousemove', (e) => {
                this.handleAttentionHover(e);
            });
            this.elements.attentionCanvas.addEventListener('mouseleave', () => {
                this.viz.clearHoverState();
                this.updateAttentionVisualization();
            });
        }
    }

    /**
     * Process input text and run forward pass
     */
    processInput() {
        if (!this.currentText.trim()) {
            this.promptTokens = [];
            this.promptTokenIds = [];
            this.currentState = null;
            this.clearVisualizations();
            return;
        }

        // Tokenize
        const { tokens, tokenIds } = this.pico.tokenize(this.currentText);
        this.promptTokens = tokens;
        this.promptTokenIds = tokenIds;

        // Build full sequence
        const allTokenIds = [...this.promptTokenIds, ...this.generatedTokenIds];
        const allTokens = [...this.promptTokens, ...this.generatedTokens];

        // Run forward pass
        this.currentState = this.pico.forward(allTokenIds, this.temperature);
        this.currentState.tokens = allTokens;

        // Update displays
        this.updateTokenCount();
        this.updateVisualizations();
    }

    /**
     * Update just the predictions (when temperature changes)
     */
    updatePredictions() {
        if (!this.currentState) return;

        // Get current token IDs
        const allTokenIds = [...this.promptTokenIds, ...this.generatedTokenIds];

        // Re-run forward with new temperature
        this.currentState = this.pico.forward(allTokenIds, this.temperature);
        this.currentState.tokens = [...this.promptTokens, ...this.generatedTokens];

        // Update probability visualization
        this.updateProbabilitiesVisualization();
    }

    /**
     * Generate one token
     */
    generateStep() {
        if (!this.currentState || !this.currentState.topK.length) return;

        // Sample from top-K with temperature
        const topK = this.currentState.topK.slice(0, this.topK);
        let selectedIdx = 0;

        // Simple weighted random selection
        const totalProb = topK.reduce((sum, item) => sum + item.probability, 0);
        let random = Math.random() * totalProb;

        for (let i = 0; i < topK.length; i++) {
            random -= topK[i].probability;
            if (random <= 0) {
                selectedIdx = i;
                break;
            }
        }

        const selectedToken = topK[selectedIdx];

        // Add to generated
        this.generatedTokens.push(selectedToken.token);
        this.generatedTokenIds.push(selectedToken.id);

        // Update current token display
        if (this.elements.currentToken) {
            this.elements.currentToken.textContent = selectedToken.token;
        }

        // Re-run forward pass with new sequence
        this.processInput();

        return selectedToken;
    }

    /**
     * Toggle auto-generation
     */
    toggleAutoGenerate() {
        if (this.isGenerating) {
            this.stopAutoGenerate();
        } else {
            this.startAutoGenerate();
        }
    }

    /**
     * Start auto-generation
     */
    startAutoGenerate() {
        this.isGenerating = true;
        if (this.elements.autoBtn) {
            this.elements.autoBtn.textContent = 'Stop';
            this.elements.autoBtn.classList.add('btn--danger');
        }

        this.generationInterval = setInterval(() => {
            const result = this.generateStep();

            // Stop after 20 tokens or end of sentence
            if (this.generatedTokens.length >= 20 ||
                result?.token === '.' ||
                result?.token === '!' ||
                result?.token === '?') {
                this.stopAutoGenerate();
            }
        }, 600);
    }

    /**
     * Stop auto-generation
     */
    stopAutoGenerate() {
        this.isGenerating = false;
        if (this.generationInterval) {
            clearInterval(this.generationInterval);
            this.generationInterval = null;
        }
        if (this.elements.autoBtn) {
            this.elements.autoBtn.textContent = 'Auto';
            this.elements.autoBtn.classList.remove('btn--danger');
        }
    }

    /**
     * Reset generation
     */
    resetGeneration() {
        this.stopAutoGenerate();
        this.generatedTokens = [];
        this.generatedTokenIds = [];
        if (this.elements.currentToken) {
            this.elements.currentToken.textContent = '-';
        }
    }

    /**
     * Update token count display
     */
    updateTokenCount() {
        if (this.elements.tokenCount) {
            const total = this.promptTokens.length + this.generatedTokens.length;
            this.elements.tokenCount.textContent = total;
        }
    }

    /**
     * Update all visualizations
     */
    updateVisualizations() {
        if (!this.currentState) return;

        this.updateEmbeddingsVisualization();
        this.updateAttentionVisualization();
        this.updateFFNVisualization();
        this.updateProbabilitiesVisualization();
        this.updateTokensVisualization();
    }

    /**
     * Clear all visualizations
     */
    clearVisualizations() {
        const canvases = [
            this.elements.embeddingsCanvas,
            this.elements.attentionCanvas,
            this.elements.ffnCanvas,
            this.elements.probabilitiesCanvas,
            this.elements.tokensCanvas
        ];

        canvases.forEach(canvas => {
            if (canvas) {
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#1e293b';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
            }
        });
    }

    /**
     * Update embeddings visualization
     */
    updateEmbeddingsVisualization() {
        if (!this.elements.embeddingsCanvas || !this.currentState) return;

        this.viz.renderEmbeddings(
            this.elements.embeddingsCanvas,
            this.currentState.tokenEmbeddings,
            this.currentState.posEmbeddings,
            this.currentState.combinedEmbeddings,
            this.currentState.tokens
        );
    }

    /**
     * Update attention visualization
     */
    updateAttentionVisualization() {
        if (!this.elements.attentionCanvas || !this.currentState) return;

        const layerState = this.currentState.layers[this.selectedLayer];
        if (!layerState) return;

        this.viz.renderAttentionHeads(
            this.elements.attentionCanvas,
            layerState.attentionWeights,
            this.currentState.tokens,
            this.selectedLayer
        );
    }

    /**
     * Update FFN visualization
     */
    updateFFNVisualization() {
        if (!this.elements.ffnCanvas || !this.currentState) return;

        const layerState = this.currentState.layers[this.selectedLayer];
        if (!layerState) return;

        this.viz.renderFFN(
            this.elements.ffnCanvas,
            layerState.ffnHidden,
            layerState.ffnOutput,
            this.currentState.tokens
        );
    }

    /**
     * Update probabilities visualization
     */
    updateProbabilitiesVisualization() {
        if (!this.elements.probabilitiesCanvas || !this.currentState) return;

        this.viz.renderProbabilities(
            this.elements.probabilitiesCanvas,
            this.currentState.topK,
            this.temperature
        );
    }

    /**
     * Update tokens visualization
     */
    updateTokensVisualization() {
        if (!this.elements.tokensCanvas) return;

        const currentIdx = this.generatedTokens.length > 0
            ? this.promptTokens.length + this.generatedTokens.length - 1
            : -1;

        this.viz.renderGeneratedTokens(
            this.elements.tokensCanvas,
            this.promptTokens,
            this.generatedTokens,
            currentIdx
        );
    }

    /**
     * Handle attention canvas hover
     */
    handleAttentionHover(event) {
        if (!this.elements.attentionCanvas || !this.currentState) return;

        const canvas = this.elements.attentionCanvas;
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // Calculate which cell was hovered
        // This is a simplified version - would need more precise calculation
        const layerState = this.currentState.layers[this.selectedLayer];
        if (!layerState || !layerState.attentionWeights) return;

        const numHeads = layerState.attentionWeights.length;
        const seqLen = this.currentState.tokens.length;

        const padding = 16;
        const titleHeight = 24;
        const tokenLabelWidth = 50;
        const headGap = 16;

        const availableWidth = canvas.width - padding * 2 - tokenLabelWidth;
        const availableHeight = canvas.height - padding * 2 - titleHeight - tokenLabelWidth;

        const headWidth = (availableWidth - headGap * (numHeads - 1)) / numHeads;
        const headHeight = availableHeight;

        // Check each head
        for (let h = 0; h < numHeads; h++) {
            const headX = padding + tokenLabelWidth + h * (headWidth + headGap);
            const headY = padding + titleHeight;

            if (x >= headX && x < headX + headWidth &&
                y >= headY && y < headY + headHeight) {

                const cellWidth = headWidth / seqLen;
                const cellHeight = headHeight / seqLen;

                const col = Math.floor((x - headX) / cellWidth);
                const row = Math.floor((y - headY) / cellHeight);

                if (row >= 0 && row < seqLen && col >= 0 && col < seqLen) {
                    this.viz.setHoverState(h, row, col);
                    this.updateAttentionVisualization();

                    // Show tooltip or update info
                    const weight = layerState.attentionWeights[h]?.[row]?.[col];
                    if (weight !== undefined && col <= row) {
                        const queryToken = this.currentState.tokens[row];
                        const keyToken = this.currentState.tokens[col];
                        console.log('Attention: "' + queryToken + '" attends to "' + keyToken + '" with weight ' + weight.toFixed(3));
                    }
                    return;
                }
            }
        }

        // No cell hovered
        this.viz.clearHoverState();
        this.updateAttentionVisualization();
    }

    /**
     * Open the explainer view
     */
    open() {
        // Cache elements first if not already done
        if (!this.elements.container) {
            this.cacheElements();
        }

        if (this.elements.container) {
            this.elements.container.style.display = 'block';
            document.body.style.overflow = 'hidden';
            this.init();

            // Resize canvases to fit
            this.resizeCanvases();
        }
    }

    /**
     * Close the explainer view
     */
    close() {
        if (this.elements.container) {
            this.elements.container.style.display = 'none';
            document.body.style.overflow = '';
        }
        this.stopAutoGenerate();
    }

    /**
     * Resize canvases to fit their containers
     */
    resizeCanvases() {
        const canvasConfigs = [
            { canvas: this.elements.embeddingsCanvas, width: 600, height: 300 },
            { canvas: this.elements.attentionCanvas, width: 700, height: 350 },
            { canvas: this.elements.ffnCanvas, width: 500, height: 300 },
            { canvas: this.elements.probabilitiesCanvas, width: 400, height: 220 },
            { canvas: this.elements.tokensCanvas, width: 800, height: 60 }
        ];

        canvasConfigs.forEach(({ canvas, width, height }) => {
            if (canvas) {
                canvas.width = width;
                canvas.height = height;
            }
        });

        // Re-render after resize
        if (this.currentState) {
            this.updateVisualizations();
        }
    }

    /**
     * Dispose resources
     */
    dispose() {
        this.stopAutoGenerate();
        this.isInitialized = false;
    }
}

// Export singleton
export const transformerExplainerFull = new TransformerExplainerFull();
