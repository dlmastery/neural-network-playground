/**
 * Transformer Visualization
 * Renders transformer architecture diagrams and attention heatmaps
 */

export class TransformerViz {
    constructor(archCanvas, attentionCanvas) {
        this.archCanvas = archCanvas;
        this.attentionCanvas = attentionCanvas;
        this.archCtx = archCanvas?.getContext('2d');
        this.attentionCtx = attentionCanvas?.getContext('2d');

        // Colors
        this.colors = {
            embedding: '#6366f1',      // Indigo
            attention: '#f59e0b',      // Amber
            ffn: '#22c55e',            // Green
            output: '#3b82f6',         // Blue
            layerNorm: '#8b5cf6',      // Purple
            text: '#f8fafc',
            textDark: '#1e293b',
            bg: '#1e293b',
            border: 'rgba(255, 255, 255, 0.1)'
        };

        // Model config for visualization
        this.modelConfig = {
            layers: 6,
            heads: 12,
            hidden: 768
        };

        // Tokens for attention visualization
        this.tokens = [];
        this.attentionWeights = null;
        this.selectedLayer = 0;
        this.selectedHead = 0;

        // Animation state
        this.animationFrame = null;
        this.dataFlowProgress = 0;
    }

    /**
     * Initialize visualization
     */
    init() {
        this.resizeCanvases();
        this.renderArchitecture();
        this.renderEmptyAttention();

        // Handle resize
        window.addEventListener('resize', () => {
            this.resizeCanvases();
            this.renderArchitecture();
            if (this.attentionWeights) {
                this.renderAttention();
            } else {
                this.renderEmptyAttention();
            }
        });
    }

    /**
     * Resize canvases to fit containers
     */
    resizeCanvases() {
        if (this.archCanvas) {
            const rect = this.archCanvas.parentElement.getBoundingClientRect();
            this.archCanvas.width = rect.width * window.devicePixelRatio;
            this.archCanvas.height = rect.height * window.devicePixelRatio;
            this.archCanvas.style.width = rect.width + 'px';
            this.archCanvas.style.height = rect.height + 'px';
            this.archCtx?.scale(window.devicePixelRatio, window.devicePixelRatio);
        }

        if (this.attentionCanvas) {
            const rect = this.attentionCanvas.parentElement.getBoundingClientRect();
            const size = Math.min(rect.width, 280);
            this.attentionCanvas.width = size * window.devicePixelRatio;
            this.attentionCanvas.height = size * window.devicePixelRatio;
            this.attentionCanvas.style.width = size + 'px';
            this.attentionCanvas.style.height = size + 'px';
            this.attentionCtx?.scale(window.devicePixelRatio, window.devicePixelRatio);
        }
    }

    /**
     * Set model configuration
     */
    setModelConfig(config) {
        this.modelConfig = { ...this.modelConfig, ...config };
        this.renderArchitecture();
    }

    /**
     * Render transformer architecture diagram
     */
    renderArchitecture() {
        if (!this.archCtx || !this.archCanvas) return;

        const ctx = this.archCtx;
        const width = this.archCanvas.width / window.devicePixelRatio;
        const height = this.archCanvas.height / window.devicePixelRatio;

        // Clear
        ctx.clearRect(0, 0, width, height);

        // Layout parameters
        const padding = 20;
        const blockWidth = 120;
        const blockHeight = 40;
        const layerHeight = 100;
        const centerX = width / 2;

        let y = padding + 20;

        // Title
        ctx.fillStyle = this.colors.text;
        ctx.font = 'bold 14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('GPT Architecture', centerX, y);
        y += 30;

        // Input tokens
        this.drawBlock(ctx, centerX - blockWidth/2, y, blockWidth, blockHeight,
            'Input Tokens', this.colors.embedding, 'white');
        y += blockHeight + 20;

        // Embeddings
        this.drawBlock(ctx, centerX - blockWidth/2, y, blockWidth, blockHeight,
            'Token + Position Emb', this.colors.embedding, 'white');
        y += blockHeight + 10;

        // Arrow
        this.drawArrow(ctx, centerX, y, centerX, y + 15);
        y += 20;

        // Transformer layers (show 3 representative)
        const showLayers = Math.min(3, this.modelConfig.layers);
        for (let i = 0; i < showLayers; i++) {
            const layerY = y;

            // Layer background
            ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
            ctx.strokeStyle = this.colors.border;
            ctx.lineWidth = 1;
            this.roundRect(ctx, centerX - blockWidth - 10, layerY - 5,
                blockWidth * 2 + 20, layerHeight - 10, 8);
            ctx.fill();
            ctx.stroke();

            // Layer label
            ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.font = '10px Inter, sans-serif';
            ctx.textAlign = 'left';
            const layerNum = i === showLayers - 1 && this.modelConfig.layers > 3
                ? this.modelConfig.layers : i + 1;
            ctx.fillText(`Layer ${layerNum}`, centerX - blockWidth - 5, layerY + 8);

            // Self-Attention
            this.drawBlock(ctx, centerX - blockWidth/2, layerY + 15, blockWidth, 28,
                `Multi-Head Attn (${this.modelConfig.heads}h)`, this.colors.attention, 'white', true);

            // Feed Forward
            this.drawBlock(ctx, centerX - blockWidth/2, layerY + 50, blockWidth, 28,
                'Feed Forward', this.colors.ffn, 'white', true);

            y += layerHeight;

            // Show ellipsis if more layers
            if (i === 1 && this.modelConfig.layers > 3) {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
                ctx.font = '16px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('⋮', centerX, y - 10);
            }
        }

        // Arrow
        this.drawArrow(ctx, centerX, y, centerX, y + 15);
        y += 20;

        // Output
        this.drawBlock(ctx, centerX - blockWidth/2, y, blockWidth, blockHeight,
            'LM Head (Softmax)', this.colors.output, 'white');
        y += blockHeight + 20;

        // Output tokens
        ctx.fillStyle = this.colors.text;
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('→ Next Token Prediction', centerX, y);

        // Animate data flow if enabled
        if (this.dataFlowProgress > 0) {
            this.renderDataFlow(ctx, width, height);
        }
    }

    /**
     * Draw a styled block
     */
    drawBlock(ctx, x, y, width, height, text, color, textColor, small = false) {
        // Gradient fill
        const gradient = ctx.createLinearGradient(x, y, x + width, y + height);
        gradient.addColorStop(0, color);
        gradient.addColorStop(1, this.adjustColor(color, -20));

        ctx.fillStyle = gradient;
        this.roundRect(ctx, x, y, width, height, 6);
        ctx.fill();

        // Text
        ctx.fillStyle = textColor;
        ctx.font = small ? '10px Inter, sans-serif' : '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, x + width/2, y + height/2);
    }

    /**
     * Draw arrow
     */
    drawArrow(ctx, x1, y1, x2, y2) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        // Arrow head
        const headLen = 6;
        const angle = Math.atan2(y2 - y1, x2 - x1);
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI/6), y2 - headLen * Math.sin(angle - Math.PI/6));
        ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI/6), y2 - headLen * Math.sin(angle + Math.PI/6));
        ctx.closePath();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.fill();
    }

    /**
     * Draw rounded rectangle
     */
    roundRect(ctx, x, y, width, height, radius) {
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
     * Adjust color brightness
     */
    adjustColor(color, amount) {
        const hex = color.replace('#', '');
        const r = Math.max(0, Math.min(255, parseInt(hex.substr(0, 2), 16) + amount));
        const g = Math.max(0, Math.min(255, parseInt(hex.substr(2, 2), 16) + amount));
        const b = Math.max(0, Math.min(255, parseInt(hex.substr(4, 2), 16) + amount));
        return `rgb(${r}, ${g}, ${b})`;
    }

    /**
     * Render data flow animation
     */
    renderDataFlow(ctx, width, height) {
        // Animated particles flowing through the network
        const progress = this.dataFlowProgress;
        const centerX = width / 2;

        ctx.fillStyle = '#22c55e';
        ctx.shadowColor = '#22c55e';
        ctx.shadowBlur = 10;

        // Draw flowing particle
        const particleY = 50 + (height - 100) * progress;
        ctx.beginPath();
        ctx.arc(centerX, particleY, 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.shadowBlur = 0;
    }

    /**
     * Set tokens for attention visualization
     */
    setTokens(tokens) {
        this.tokens = tokens;
        if (!this.attentionWeights) {
            this.renderEmptyAttention();
        }
    }

    /**
     * Set attention weights and render
     * @param {Float32Array[][]} weights - [layers][heads][tokens][tokens]
     */
    setAttentionWeights(weights) {
        this.attentionWeights = weights;
        this.renderAttention();
    }

    /**
     * Set selected layer and head
     */
    setSelection(layer, head) {
        this.selectedLayer = layer;
        this.selectedHead = head;
        if (this.attentionWeights) {
            this.renderAttention();
        }
    }

    /**
     * Render empty attention placeholder
     */
    renderEmptyAttention() {
        if (!this.attentionCtx || !this.attentionCanvas) return;

        const ctx = this.attentionCtx;
        const size = this.attentionCanvas.width / window.devicePixelRatio;

        ctx.clearRect(0, 0, size, size);

        // Background
        ctx.fillStyle = 'rgba(30, 41, 59, 0.5)';
        ctx.fillRect(0, 0, size, size);

        // Placeholder text
        ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Generate text to see', size/2, size/2 - 10);
        ctx.fillText('attention patterns', size/2, size/2 + 10);
    }

    /**
     * Render attention heatmap
     */
    renderAttention() {
        if (!this.attentionCtx || !this.attentionCanvas || !this.attentionWeights) return;

        const ctx = this.attentionCtx;
        const size = this.attentionCanvas.width / window.devicePixelRatio;
        const numTokens = this.tokens.length;

        if (numTokens === 0) {
            this.renderEmptyAttention();
            return;
        }

        ctx.clearRect(0, 0, size, size);

        // Get attention matrix for selected layer/head
        const layerAttn = this.attentionWeights[this.selectedLayer];
        if (!layerAttn) {
            this.renderEmptyAttention();
            return;
        }

        const headAttn = layerAttn[this.selectedHead];
        if (!headAttn) {
            this.renderEmptyAttention();
            return;
        }

        // Calculate cell size
        const padding = 30;
        const gridSize = size - padding * 2;
        const cellSize = gridSize / numTokens;

        // Draw heatmap
        for (let i = 0; i < numTokens; i++) {
            for (let j = 0; j < numTokens; j++) {
                const value = headAttn[i * numTokens + j] || 0;
                const x = padding + j * cellSize;
                const y = padding + i * cellSize;

                // Color based on attention weight
                const color = this.getAttentionColor(value);
                ctx.fillStyle = color;
                ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
            }
        }

        // Draw labels if space allows
        if (cellSize > 15) {
            ctx.fillStyle = this.colors.text;
            ctx.font = '9px JetBrains Mono, monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            for (let i = 0; i < numTokens; i++) {
                // Truncate token for display
                const token = this.tokens[i].slice(0, 3);

                // Top labels (keys)
                ctx.save();
                ctx.translate(padding + i * cellSize + cellSize/2, padding - 10);
                ctx.rotate(-Math.PI/4);
                ctx.fillText(token, 0, 0);
                ctx.restore();

                // Left labels (queries)
                ctx.fillText(token, padding - 15, padding + i * cellSize + cellSize/2);
            }
        }

        // Color legend
        this.drawColorLegend(ctx, size);
    }

    /**
     * Get color for attention value
     */
    getAttentionColor(value) {
        // Use a warm color scale (yellow -> orange -> red)
        const v = Math.max(0, Math.min(1, value));

        if (v < 0.5) {
            // Interpolate from dark blue to yellow
            const t = v * 2;
            const r = Math.round(30 + t * (250 - 30));
            const g = Math.round(41 + t * (200 - 41));
            const b = Math.round(59 + t * (50 - 59));
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            // Interpolate from yellow to red
            const t = (v - 0.5) * 2;
            const r = Math.round(250 + t * (239 - 250));
            const g = Math.round(200 - t * 200);
            const b = Math.round(50 - t * 18);
            return `rgb(${r}, ${g}, ${b})`;
        }
    }

    /**
     * Draw color legend
     */
    drawColorLegend(ctx, size) {
        const legendWidth = 60;
        const legendHeight = 10;
        const x = size - legendWidth - 10;
        const y = size - legendHeight - 10;

        // Draw gradient
        const gradient = ctx.createLinearGradient(x, y, x + legendWidth, y);
        gradient.addColorStop(0, this.getAttentionColor(0));
        gradient.addColorStop(0.5, this.getAttentionColor(0.5));
        gradient.addColorStop(1, this.getAttentionColor(1));

        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, legendWidth, legendHeight);

        // Labels
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.font = '8px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('0', x, y - 3);
        ctx.textAlign = 'right';
        ctx.fillText('1', x + legendWidth, y - 3);
    }

    /**
     * Generate mock attention weights for visualization demo
     */
    generateMockAttention(numTokens) {
        const layers = this.modelConfig.layers;
        const heads = this.modelConfig.heads;
        const weights = [];

        for (let l = 0; l < layers; l++) {
            const layerWeights = [];
            for (let h = 0; h < heads; h++) {
                // Generate random attention pattern with causal mask
                const attn = new Float32Array(numTokens * numTokens);
                for (let i = 0; i < numTokens; i++) {
                    let sum = 0;
                    for (let j = 0; j <= i; j++) {
                        // Random weights with some structure
                        const localBias = Math.exp(-Math.abs(i - j) * 0.5);
                        attn[i * numTokens + j] = Math.random() * localBias;
                        sum += attn[i * numTokens + j];
                    }
                    // Normalize (softmax-like)
                    for (let j = 0; j <= i; j++) {
                        attn[i * numTokens + j] /= sum;
                    }
                }
                layerWeights.push(attn);
            }
            weights.push(layerWeights);
        }

        return weights;
    }

    /**
     * Start data flow animation
     */
    animateDataFlow() {
        this.dataFlowProgress = 0;
        const animate = () => {
            this.dataFlowProgress += 0.02;
            this.renderArchitecture();

            if (this.dataFlowProgress < 1) {
                this.animationFrame = requestAnimationFrame(animate);
            } else {
                this.dataFlowProgress = 0;
                this.renderArchitecture();
            }
        };
        animate();
    }

    /**
     * Stop animations
     */
    stopAnimation() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        this.dataFlowProgress = 0;
    }

    /**
     * Dispose resources
     */
    dispose() {
        this.stopAnimation();
    }
}
