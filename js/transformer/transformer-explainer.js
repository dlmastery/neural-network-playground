/**
 * Transformer Explainer
 * Educational modal with interactive demonstrations of transformer concepts
 */

import { PicoTransformer, TransformerConcepts } from './pico-transformer.js';

export class TransformerExplainer {
    constructor(model) {
        this.model = model;
        this.isInitialized = false;
        this.currentTab = 't-overview';

        // Educational mini-transformer for demonstrations
        this.picoTransformer = new PicoTransformer({
            vocabSize: 1000,
            contextLength: 32,
            embeddingDim: 64,
            numHeads: 4,
            numLayers: 2,
            ffnDim: 256
        });

        // Cache elements
        this.elements = {};

        // Animation frames
        this.animationFrames = {};

        // Demo state
        this.demoTokens = [464, 2068, 7586, 21831]; // "The quick brown fox" (mock IDs)
        this.attentionState = null;
    }

    /**
     * Initialize the explainer
     */
    init() {
        if (this.isInitialized) return;

        this.cacheElements();
        this.setupTabSwitching();
        this.setupTokenizerDemo();
        this.setupGenerationDemo();
        this.renderOverviewDiagram();
        this.renderEmbeddingVisuals();
        this.renderAttentionDemo();
        this.renderMultiHeadDemo();

        // Run initial forward pass for demo data
        this.runPicoDemo();

        this.isInitialized = true;
    }

    /**
     * Run the PicoTransformer for demonstration
     */
    runPicoDemo() {
        // Set up callbacks to capture intermediate states
        this.picoTransformer.onAttention = (layerIdx, weights, tokens) => {
            this.attentionState = { layerIdx, weights, tokens };
        };

        // Run forward pass
        const state = this.picoTransformer.forward(this.demoTokens);

        // Update visualizations with real computed values
        if (state.attentionWeights.length > 0) {
            this.updateAttentionVisualization(state.attentionWeights[0]);
        }
    }

    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements = {
            // Tab buttons
            tabBtns: document.querySelectorAll('#transformer-explainer-modal .explainer-tab-btn'),

            // Tab panels
            overviewPanel: document.getElementById('t-overview-panel'),
            tokenizationPanel: document.getElementById('t-tokenization-panel'),
            embeddingsPanel: document.getElementById('t-embeddings-panel'),
            attentionPanel: document.getElementById('t-attention-panel'),
            multiheadPanel: document.getElementById('t-multihead-panel'),
            generationPanel: document.getElementById('t-generation-panel'),

            // Tokenizer demo
            tokenizerInput: document.getElementById('tokenizer-input'),
            tokenizerOutput: document.getElementById('tokenizer-output'),
            tokenizerIds: document.getElementById('tokenizer-ids'),

            // Canvases
            overviewCanvas: document.getElementById('transformer-overview-canvas'),
            tokenEmbCanvas: document.getElementById('embedding-token-canvas'),
            posEmbCanvas: document.getElementById('embedding-position-canvas'),
            combinedEmbCanvas: document.getElementById('embedding-combined-canvas'),
            qCanvas: document.getElementById('attention-q-canvas'),
            kCanvas: document.getElementById('attention-k-canvas'),
            scoresCanvas: document.getElementById('attention-scores-canvas'),
            vCanvas: document.getElementById('attention-v-canvas'),
            multiheadCanvas: document.getElementById('multihead-canvas'),

            // Generation demo
            generationViz: document.getElementById('generation-step-viz'),
            stepBtn: document.getElementById('generation-step-btn'),
            autoBtn: document.getElementById('generation-auto-btn')
        };
    }

    /**
     * Setup tab switching
     */
    setupTabSwitching() {
        this.elements.tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                this.switchTab(tab);
            });
        });
    }

    /**
     * Switch to a tab
     */
    switchTab(tabId) {
        // Update buttons
        this.elements.tabBtns.forEach(btn => {
            btn.classList.toggle('explainer-tab-btn--active', btn.dataset.tab === tabId);
        });

        // Update panels
        const panels = document.querySelectorAll('#transformer-explainer-modal .explainer-tab-panel');
        panels.forEach(panel => {
            const isActive = panel.id === `${tabId}-panel`;
            panel.classList.toggle('explainer-tab-panel--active', isActive);
        });

        this.currentTab = tabId;
    }

    /**
     * Setup tokenizer demo
     */
    setupTokenizerDemo() {
        if (!this.elements.tokenizerInput) return;

        const updateTokens = () => {
            const text = this.elements.tokenizerInput.value;
            this.renderTokens(text);
        };

        this.elements.tokenizerInput.addEventListener('input', updateTokens);
        updateTokens(); // Initial render
    }

    /**
     * Render tokens for the tokenizer demo
     */
    renderTokens(text) {
        if (!this.elements.tokenizerOutput || !this.elements.tokenizerIds) return;

        // Clear previous content safely
        while (this.elements.tokenizerOutput.firstChild) {
            this.elements.tokenizerOutput.removeChild(this.elements.tokenizerOutput.firstChild);
        }

        if (!text) {
            this.elements.tokenizerIds.textContent = 'IDs: []';
            return;
        }

        // If model is loaded, use real tokenization
        if (this.model?.isLoaded) {
            const { tokens, ids } = this.model.tokenize(text);

            tokens.forEach(token => {
                const span = document.createElement('span');
                span.className = 'token';
                span.textContent = token;
                this.elements.tokenizerOutput.appendChild(span);
            });

            this.elements.tokenizerIds.textContent = `IDs: [${ids.join(', ')}]`;
        } else {
            // Mock tokenization by splitting on spaces and subwords
            const mockTokens = this.mockTokenize(text);

            mockTokens.forEach((token, i) => {
                const span = document.createElement('span');
                span.className = 'token';
                span.textContent = token;
                this.elements.tokenizerOutput.appendChild(span);
            });

            const mockIds = mockTokens.map((_, i) => Math.floor(Math.random() * 50000));
            this.elements.tokenizerIds.textContent = `IDs: [${mockIds.join(', ')}] (mock - load model for real IDs)`;
        }
    }

    /**
     * Mock tokenization for demo when model not loaded
     */
    mockTokenize(text) {
        const tokens = [];

        // Split by spaces, keeping punctuation separate
        const parts = text.split(/(\s+|[.,!?;:'"()-])/);

        for (const part of parts) {
            if (!part) continue;

            // Keep whitespace and punctuation as single tokens
            if (/^\s+$/.test(part) || /^[.,!?;:'"()-]$/.test(part)) {
                tokens.push(part);
                continue;
            }

            // Split long words into subwords (BPE-like behavior)
            if (part.length > 4) {
                // Common prefixes/suffixes
                if (part.endsWith('ing')) {
                    tokens.push(part.slice(0, -3));
                    tokens.push('ing');
                } else if (part.endsWith('ed')) {
                    tokens.push(part.slice(0, -2));
                    tokens.push('ed');
                } else if (part.endsWith('ly')) {
                    tokens.push(part.slice(0, -2));
                    tokens.push('ly');
                } else if (part.endsWith('tion')) {
                    tokens.push(part.slice(0, -4));
                    tokens.push('tion');
                } else {
                    tokens.push(part);
                }
            } else {
                tokens.push(part);
            }
        }

        return tokens.filter(t => t);
    }

    /**
     * Render overview diagram
     */
    renderOverviewDiagram() {
        const canvas = this.elements.overviewCanvas;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        ctx.clearRect(0, 0, width, height);

        // Background
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const centerX = width / 2;
        let y = 30;

        // Title
        ctx.fillStyle = '#1e293b';
        ctx.font = 'bold 14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Decoder-Only Transformer (GPT)', centerX, y);
        y += 30;

        // Input
        this.drawExplainerBlock(ctx, centerX - 60, y, 120, 30, 'Input Tokens', '#6366f1');
        y += 40;
        this.drawArrow(ctx, centerX, y - 5, centerX, y + 5, '#94a3b8');
        y += 15;

        // Embedding
        this.drawExplainerBlock(ctx, centerX - 80, y, 160, 30, 'Token + Position Embedding', '#8b5cf6');
        y += 40;
        this.drawArrow(ctx, centerX, y - 5, centerX, y + 5, '#94a3b8');
        y += 15;

        // Transformer block
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        this.roundRect(ctx, centerX - 100, y, 200, 100);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#64748b';
        ctx.font = '10px Inter, sans-serif';
        ctx.fillText('×N Layers', centerX + 90, y + 15);

        // Self-Attention
        this.drawExplainerBlock(ctx, centerX - 70, y + 15, 140, 25, 'Masked Self-Attention', '#f59e0b');
        // FFN
        this.drawExplainerBlock(ctx, centerX - 70, y + 55, 140, 25, 'Feed-Forward Network', '#22c55e');

        y += 110;
        this.drawArrow(ctx, centerX, y - 5, centerX, y + 5, '#94a3b8');
        y += 15;

        // Output
        this.drawExplainerBlock(ctx, centerX - 60, y, 120, 30, 'Output Logits', '#3b82f6');
        y += 40;

        // Softmax
        ctx.fillStyle = '#64748b';
        ctx.font = '12px Inter, sans-serif';
        ctx.fillText('↓ Softmax → Next Token', centerX, y);
    }

    /**
     * Draw a styled block for explainer
     */
    drawExplainerBlock(ctx, x, y, width, height, text, color) {
        // Background
        ctx.fillStyle = color;
        this.roundRect(ctx, x, y, width, height, 4);
        ctx.fill();

        // Text
        ctx.fillStyle = 'white';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, x + width/2, y + height/2);
    }

    /**
     * Draw arrow
     */
    drawArrow(ctx, x1, y1, x2, y2, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        // Arrow head
        const headLen = 5;
        const angle = Math.atan2(y2 - y1, x2 - x1);
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI/6), y2 - headLen * Math.sin(angle - Math.PI/6));
        ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI/6), y2 - headLen * Math.sin(angle + Math.PI/6));
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
    }

    /**
     * Draw rounded rectangle path
     */
    roundRect(ctx, x, y, width, height, radius = 4) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
    }

    /**
     * Render embedding visualizations
     */
    renderEmbeddingVisuals() {
        this.renderEmbeddingCanvas(this.elements.tokenEmbCanvas, 'Token', '#6366f1');
        this.renderEmbeddingCanvas(this.elements.posEmbCanvas, 'Position', '#8b5cf6');
        this.renderEmbeddingCanvas(this.elements.combinedEmbCanvas, 'Combined', '#3b82f6');
    }

    /**
     * Render a single embedding visualization
     */
    renderEmbeddingCanvas(canvas, label, color) {
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const size = canvas.width;

        ctx.clearRect(0, 0, size, size);

        // Background
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, size, size);

        // Draw a "vector" representation
        const vectorWidth = 30;
        const startX = (size - vectorWidth) / 2;
        const startY = 30;
        const vectorHeight = size - 60;
        const numElements = 8;
        const elementHeight = vectorHeight / numElements;

        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillStyle = '#64748b';
        ctx.fillText('768-dim', size/2, 20);

        for (let i = 0; i < numElements; i++) {
            const value = Math.random();
            const y = startY + i * elementHeight;

            // Color based on value
            const r = Math.round(99 + (1 - value) * 100);
            const g = Math.round(102 + (1 - value) * 100);
            const b = Math.round(241);
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.3 + value * 0.7})`;

            ctx.fillRect(startX, y, vectorWidth, elementHeight - 2);

            // Value text
            ctx.fillStyle = '#1e293b';
            ctx.font = '9px JetBrains Mono, monospace';
            ctx.textAlign = 'left';
            ctx.fillText(value.toFixed(2), startX + vectorWidth + 5, y + elementHeight/2 + 3);
        }

        // Ellipsis
        ctx.fillStyle = '#64748b';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('...', size/2, size - 15);
    }

    /**
     * Render attention QKV demo
     */
    renderAttentionDemo() {
        this.renderMatrixCanvas(this.elements.qCanvas, 'Q');
        this.renderMatrixCanvas(this.elements.kCanvas, 'K');
        this.renderMatrixCanvas(this.elements.scoresCanvas, 'Scores', true);
        this.renderMatrixCanvas(this.elements.vCanvas, 'V');
    }

    /**
     * Render a matrix visualization
     */
    renderMatrixCanvas(canvas, label, isScores = false) {
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const size = canvas.width;

        ctx.clearRect(0, 0, size, size);

        // Background
        ctx.fillStyle = '#f1f5f9';
        ctx.fillRect(0, 0, size, size);

        const gridSize = 4;
        const cellSize = (size - 20) / gridSize;
        const startX = 10;
        const startY = 10;

        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                const x = startX + j * cellSize;
                const y = startY + i * cellSize;

                let value;
                if (isScores) {
                    // Attention scores with causal mask
                    value = j <= i ? Math.random() : 0;
                    // Normalize row
                    if (j === i) {
                        value = 0.5 + Math.random() * 0.3; // Higher diagonal
                    }
                } else {
                    value = Math.random();
                }

                // Color
                if (isScores && j > i) {
                    ctx.fillStyle = '#e2e8f0'; // Masked
                } else {
                    const intensity = Math.round(value * 255);
                    ctx.fillStyle = isScores
                        ? `rgb(${255 - intensity}, ${200 - intensity * 0.5}, ${50})`
                        : `rgba(99, 102, 241, ${0.2 + value * 0.8})`;
                }

                ctx.fillRect(x, y, cellSize - 2, cellSize - 2);
            }
        }
    }

    /**
     * Render multi-head attention demo
     */
    renderMultiHeadDemo() {
        const canvas = this.elements.multiheadCanvas;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        ctx.clearRect(0, 0, width, height);

        // Background
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, width, height);

        const numHeads = 4; // Show 4 heads for simplicity
        const headWidth = 80;
        const headHeight = 60;
        const startX = (width - numHeads * headWidth - (numHeads - 1) * 20) / 2;
        const y = 80;

        // Input
        ctx.fillStyle = '#6366f1';
        this.roundRect(ctx, width/2 - 50, 20, 100, 30, 4);
        ctx.fill();
        ctx.fillStyle = 'white';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Input', width/2, 40);

        // Draw arrows to heads
        for (let i = 0; i < numHeads; i++) {
            const headX = startX + i * (headWidth + 20);
            const headCenterX = headX + headWidth/2;

            // Arrow from input to head
            ctx.strokeStyle = '#94a3b8';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(width/2, 50);
            ctx.lineTo(headCenterX, y);
            ctx.stroke();
        }

        // Draw heads
        const headColors = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6'];
        for (let i = 0; i < numHeads; i++) {
            const headX = startX + i * (headWidth + 20);

            ctx.fillStyle = headColors[i];
            this.roundRect(ctx, headX, y, headWidth, headHeight, 4);
            ctx.fill();

            ctx.fillStyle = 'white';
            ctx.font = '10px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`Head ${i + 1}`, headX + headWidth/2, y + 25);
            ctx.font = '9px Inter, sans-serif';
            ctx.fillText('64-dim', headX + headWidth/2, y + 40);
        }

        // Concat box
        const concatY = y + headHeight + 40;
        ctx.fillStyle = '#8b5cf6';
        this.roundRect(ctx, width/2 - 80, concatY, 160, 30, 4);
        ctx.fill();
        ctx.fillStyle = 'white';
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText('Concatenate (256-dim)', width/2, concatY + 20);

        // Arrows to concat
        for (let i = 0; i < numHeads; i++) {
            const headX = startX + i * (headWidth + 20);
            const headCenterX = headX + headWidth/2;

            ctx.strokeStyle = '#94a3b8';
            ctx.beginPath();
            ctx.moveTo(headCenterX, y + headHeight);
            ctx.lineTo(width/2, concatY);
            ctx.stroke();
        }

        // Linear projection
        const linearY = concatY + 50;
        ctx.fillStyle = '#64748b';
        this.roundRect(ctx, width/2 - 60, linearY, 120, 30, 4);
        ctx.fill();
        ctx.fillStyle = 'white';
        ctx.fillText('Linear (768-dim)', width/2, linearY + 20);

        // Arrow to linear
        ctx.strokeStyle = '#94a3b8';
        ctx.beginPath();
        ctx.moveTo(width/2, concatY + 30);
        ctx.lineTo(width/2, linearY);
        ctx.stroke();

        // Output
        ctx.fillStyle = '#1e293b';
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText('Output', width/2, linearY + 50);
    }

    /**
     * Setup generation demo
     */
    setupGenerationDemo() {
        if (!this.elements.stepBtn || !this.elements.autoBtn) return;

        let tokens = ['The', ' cat', ' sat'];
        let isAutoGenerating = false;

        const renderStep = () => {
            if (!this.elements.generationViz) return;

            // Clear safely
            while (this.elements.generationViz.firstChild) {
                this.elements.generationViz.removeChild(this.elements.generationViz.firstChild);
            }

            // Render tokens
            tokens.forEach((token, i) => {
                const span = document.createElement('span');
                span.style.cssText = `
                    display: inline-block;
                    padding: 4px 8px;
                    margin: 2px;
                    background: ${i < tokens.length - 1 ? '#dbeafe' : '#dcfce7'};
                    border-radius: 4px;
                    font-family: 'JetBrains Mono', monospace;
                `;
                span.textContent = token;
                this.elements.generationViz.appendChild(span);
            });

            // Add cursor
            const cursor = document.createElement('span');
            cursor.style.cssText = `
                display: inline-block;
                width: 2px;
                height: 16px;
                background: #6366f1;
                margin-left: 4px;
                animation: blink 1s infinite;
            `;
            this.elements.generationViz.appendChild(cursor);
        };

        const nextTokens = [' on', ' the', ' mat', '.', ' It', ' was', ' warm', '.'];
        let tokenIndex = 0;

        const addToken = () => {
            if (tokenIndex < nextTokens.length) {
                tokens.push(nextTokens[tokenIndex]);
                tokenIndex++;
                renderStep();
                return true;
            }
            return false;
        };

        this.elements.stepBtn.addEventListener('click', () => {
            if (!addToken()) {
                // Reset
                tokens = ['The', ' cat', ' sat'];
                tokenIndex = 0;
                renderStep();
            }
        });

        this.elements.autoBtn.addEventListener('click', () => {
            if (isAutoGenerating) {
                isAutoGenerating = false;
                this.elements.autoBtn.textContent = 'Auto-Generate';
                return;
            }

            isAutoGenerating = true;
            this.elements.autoBtn.textContent = 'Stop';

            const autoGenerate = () => {
                if (!isAutoGenerating) return;

                if (addToken()) {
                    setTimeout(autoGenerate, 500);
                } else {
                    isAutoGenerating = false;
                    this.elements.autoBtn.textContent = 'Auto-Generate';
                }
            };

            autoGenerate();
        });

        renderStep();
    }

    /**
     * Update attention visualization with real computed values from PicoTransformer
     */
    updateAttentionVisualization(headWeights) {
        if (!headWeights || headWeights.length === 0) return;

        // Update the scores canvas with real attention weights from head 0
        const scoresCanvas = this.elements.scoresCanvas;
        if (!scoresCanvas) return;

        const ctx = scoresCanvas.getContext('2d');
        const size = scoresCanvas.width;
        const weights = headWeights[0]; // First head

        ctx.clearRect(0, 0, size, size);

        // Background
        ctx.fillStyle = '#f1f5f9';
        ctx.fillRect(0, 0, size, size);

        const seqLen = this.demoTokens.length;
        const cellSize = (size - 20) / seqLen;
        const startX = 10;
        const startY = 10;

        // Draw attention heatmap
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < seqLen; j++) {
                const x = startX + j * cellSize;
                const y = startY + i * cellSize;
                const value = weights[i * seqLen + j];

                // Color based on attention weight (causal mask applied)
                if (j > i) {
                    ctx.fillStyle = '#e2e8f0'; // Masked (future tokens)
                } else {
                    // Yellow to red scale
                    const intensity = Math.min(1, value * 2); // Boost visibility
                    const r = Math.round(255);
                    const g = Math.round(200 - intensity * 150);
                    const b = Math.round(50);
                    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.3 + intensity * 0.7})`;
                }

                ctx.fillRect(x, y, cellSize - 2, cellSize - 2);
            }
        }

        // Add labels
        const labels = ['The', 'quick', 'brown', 'fox'];
        ctx.fillStyle = '#1e293b';
        ctx.font = '8px Inter, sans-serif';
        ctx.textAlign = 'center';

        for (let i = 0; i < Math.min(seqLen, labels.length); i++) {
            // Column labels (Keys)
            ctx.fillText(labels[i].slice(0, 3), startX + i * cellSize + cellSize/2, 8);
            // Row labels (Queries)
            ctx.fillText(labels[i].slice(0, 3), 8, startY + i * cellSize + cellSize/2 + 3);
        }
    }

    /**
     * Dispose resources
     */
    dispose() {
        Object.values(this.animationFrames).forEach(frame => {
            cancelAnimationFrame(frame);
        });
    }
}
