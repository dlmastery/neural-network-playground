/**
 * Transformer Explainer Visualizations
 * Canvas-based rendering functions for the interactive Transformer Explainer
 */

export class ExplainerViz {
    constructor() {
        // Color schemes
        this.colors = {
            // Embeddings
            tokenEmb: { low: '#1e1b4b', high: '#6366f1' },
            posEmb: { low: '#4c1d95', high: '#8b5cf6' },
            combined: { low: '#1e3a5f', high: '#3b82f6' },

            // Attention
            attentionLow: '#fef3c7',
            attentionHigh: '#ef4444',
            masked: '#374151',

            // FFN
            ffnActivation: '#22c55e',

            // Probability
            probability: '#6366f1',

            // UI
            background: '#0f172a',
            surface: '#1e293b',
            text: '#f8fafc',
            textMuted: '#94a3b8',
            border: '#334155',
            highlight: '#fbbf24'
        };

        // Interaction state
        this.hoverState = {
            headIdx: null,
            row: null,
            col: null
        };
    }

    /**
     * Render embedding columns (token, positional, combined)
     */
    renderEmbeddings(canvas, tokenEmb, posEmb, combinedEmb, tokens) {
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = this.colors.surface;
        ctx.fillRect(0, 0, width, height);

        const padding = 16;
        const tokenLabelHeight = 30;
        const columnGap = 16;
        const sectionGap = 24;

        const seqLen = tokens.length;
        const embDim = tokenEmb[0]?.length || 64;

        // Calculate column dimensions
        const totalSectionWidth = (width - padding * 2 - sectionGap * 2) / 3;
        const colWidth = Math.min(40, (totalSectionWidth - columnGap * (seqLen - 1)) / seqLen);

        // Section titles
        const sections = [
            { title: 'Token Embeddings', data: tokenEmb, color: this.colors.tokenEmb },
            { title: 'Positional Enc.', data: posEmb, color: this.colors.posEmb },
            { title: 'Combined', data: combinedEmb, color: this.colors.combined }
        ];

        sections.forEach((section, sectionIdx) => {
            const sectionX = padding + sectionIdx * (totalSectionWidth + sectionGap);

            // Section title
            ctx.fillStyle = this.colors.text;
            ctx.font = 'bold 11px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(section.title, sectionX + totalSectionWidth / 2, padding + 12);

            // Draw plus/equals signs
            if (sectionIdx === 1) {
                ctx.fillStyle = this.colors.textMuted;
                ctx.font = 'bold 20px Inter, sans-serif';
                ctx.fillText('+', sectionX - sectionGap / 2, height / 2);
            } else if (sectionIdx === 2) {
                ctx.fillText('=', sectionX - sectionGap / 2, height / 2);
            }

            // Draw each token's embedding column
            for (let t = 0; t < seqLen; t++) {
                const colX = sectionX + t * (colWidth + columnGap);
                const colY = padding + tokenLabelHeight + 20;
                const colHeight = height - colY - padding - 30;

                // Token label
                ctx.fillStyle = this.colors.text;
                ctx.font = '10px JetBrains Mono, monospace';
                ctx.textAlign = 'center';
                const label = tokens[t].length > 5 ? tokens[t].slice(0, 4) + '..' : tokens[t];
                ctx.fillText(label, colX + colWidth / 2, colY - 6);

                // Draw embedding values as vertical heatmap
                if (section.data && section.data[t]) {
                    this.renderEmbeddingColumn(
                        ctx, section.data[t],
                        colX, colY, colWidth, colHeight,
                        section.color
                    );
                }
            }

            // Dimension label
            ctx.fillStyle = this.colors.textMuted;
            ctx.font = '9px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(embDim + '-dim', sectionX + totalSectionWidth / 2, height - padding);
        });
    }

    /**
     * Render a single embedding column as vertical heatmap
     */
    renderEmbeddingColumn(ctx, embedding, x, y, width, height, colorScheme) {
        const numValues = embedding.length;
        const cellHeight = height / Math.min(numValues, 32); // Show max 32 cells
        const step = Math.ceil(numValues / 32);

        // Find min/max for normalization
        let min = Infinity, max = -Infinity;
        for (const v of embedding) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        const range = max - min || 1;

        for (let i = 0; i < Math.min(numValues, 32); i++) {
            const idx = i * step;
            const value = embedding[idx];
            const normalized = (value - min) / range;

            const cellY = y + i * cellHeight;

            // Interpolate color
            ctx.fillStyle = this.interpolateColor(
                colorScheme.low,
                colorScheme.high,
                normalized
            );
            ctx.fillRect(x, cellY, width, cellHeight - 1);
        }

        // Border
        ctx.strokeStyle = this.colors.border;
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, width, height);
    }

    /**
     * Render multi-head attention grid
     */
    renderAttentionHeads(canvas, attentionWeights, tokens, selectedLayer = 0, hoverCallback = null) {
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = this.colors.surface;
        ctx.fillRect(0, 0, width, height);

        if (!attentionWeights || attentionWeights.length === 0) return;

        const numHeads = attentionWeights.length;
        const seqLen = tokens.length;

        const padding = 16;
        const titleHeight = 24;
        const tokenLabelWidth = 50;
        const headGap = 16;

        // Calculate head dimensions
        const availableWidth = width - padding * 2 - tokenLabelWidth;
        const availableHeight = height - padding * 2 - titleHeight - tokenLabelWidth;

        const headWidth = (availableWidth - headGap * (numHeads - 1)) / numHeads;
        const headHeight = availableHeight;

        // Title
        ctx.fillStyle = this.colors.text;
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Multi-Head Self-Attention (Layer ' + (selectedLayer + 1) + ')', width / 2, padding + 12);

        // Draw each head
        for (let h = 0; h < numHeads; h++) {
            const headX = padding + tokenLabelWidth + h * (headWidth + headGap);
            const headY = padding + titleHeight;

            // Head label
            ctx.fillStyle = this.colors.textMuted;
            ctx.font = '10px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Head ' + (h + 1), headX + headWidth / 2, headY - 4);

            // Draw attention heatmap
            this.renderAttentionHeatmap(
                ctx, attentionWeights[h], tokens,
                headX, headY, headWidth, headHeight,
                h, hoverCallback
            );
        }

        // Token labels (rows - queries)
        ctx.fillStyle = this.colors.text;
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'right';

        const cellHeight = headHeight / seqLen;
        for (let i = 0; i < seqLen; i++) {
            const label = tokens[i].length > 6 ? tokens[i].slice(0, 5) + '..' : tokens[i];
            const labelY = padding + titleHeight + i * cellHeight + cellHeight / 2 + 3;
            ctx.fillText(label, padding + tokenLabelWidth - 4, labelY);
        }

        // Formula
        ctx.fillStyle = this.colors.textMuted;
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V', width / 2, height - 8);
    }

    /**
     * Render single attention heatmap
     */
    renderAttentionHeatmap(ctx, weights, tokens, x, y, width, height, headIdx, hoverCallback) {
        const seqLen = tokens.length;
        const cellWidth = width / seqLen;
        const cellHeight = height / seqLen;

        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < seqLen; j++) {
                const cellX = x + j * cellWidth;
                const cellY = y + i * cellHeight;

                // Masked (future tokens)
                if (j > i) {
                    ctx.fillStyle = this.colors.masked;
                } else {
                    // Get attention weight
                    const weight = weights[i]?.[j] ?? 0;
                    ctx.fillStyle = this.interpolateColor(
                        this.colors.attentionLow,
                        this.colors.attentionHigh,
                        Math.min(1, weight * 2) // Boost for visibility
                    );
                }

                ctx.fillRect(cellX, cellY, cellWidth - 1, cellHeight - 1);

                // Highlight on hover
                if (this.hoverState.headIdx === headIdx &&
                    this.hoverState.row === i &&
                    this.hoverState.col === j) {
                    ctx.strokeStyle = this.colors.highlight;
                    ctx.lineWidth = 2;
                    ctx.strokeRect(cellX, cellY, cellWidth - 1, cellHeight - 1);
                }
            }
        }

        // Border
        ctx.strokeStyle = this.colors.border;
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, width, height);
    }

    /**
     * Render FFN activations
     */
    renderFFN(canvas, ffnHidden, ffnOutput, tokens) {
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = this.colors.surface;
        ctx.fillRect(0, 0, width, height);

        if (!ffnHidden || ffnHidden.length === 0) return;

        const padding = 16;
        const titleHeight = 30;

        // Title
        ctx.fillStyle = this.colors.text;
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Feed-Forward Network', width / 2, padding + 12);

        // Formula
        ctx.fillStyle = this.colors.textMuted;
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.fillText('FFN(x) = GELU(xW1 + b1)W2 + b2', width / 2, padding + 26);

        // Draw hidden layer activations for last token
        const lastIdx = ffnHidden.length - 1;
        const hidden = ffnHidden[lastIdx];
        const output = ffnOutput ? ffnOutput[lastIdx] : null;

        if (!hidden) return;

        const barHeight = 6;
        const barGap = 2;
        const numBars = Math.min(32, hidden.length);
        const totalBarHeight = numBars * (barHeight + barGap);

        const startY = padding + titleHeight + 20;
        const maxBarWidth = (width - padding * 4) / 2;

        // Find max for normalization
        let maxVal = 0;
        for (const v of hidden) {
            if (Math.abs(v) > maxVal) maxVal = Math.abs(v);
        }
        maxVal = maxVal || 1;

        // Section labels
        ctx.fillStyle = this.colors.text;
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Hidden (256-dim)', padding + maxBarWidth / 2, startY - 6);

        if (output) {
            ctx.fillText('Output (64-dim)', width - padding - maxBarWidth / 2, startY - 6);
        }

        // Draw hidden layer bars
        const step = Math.ceil(hidden.length / numBars);
        for (let i = 0; i < numBars; i++) {
            const idx = i * step;
            const value = hidden[idx];
            const normalized = Math.abs(value) / maxVal;
            const barWidth = normalized * maxBarWidth;

            const barY = startY + i * (barHeight + barGap);

            // Background
            ctx.fillStyle = this.colors.border;
            ctx.fillRect(padding, barY, maxBarWidth, barHeight);

            // Activation bar
            ctx.fillStyle = value >= 0 ? '#22c55e' : '#ef4444';
            ctx.fillRect(padding, barY, barWidth, barHeight);
        }

        // Arrow between sections
        ctx.fillStyle = this.colors.textMuted;
        ctx.font = '16px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('->', width / 2, startY + totalBarHeight / 2);
        ctx.font = '9px Inter, sans-serif';
        ctx.fillText('GELU', width / 2, startY + totalBarHeight / 2 + 14);

        // Draw output layer bars
        if (output) {
            let outMax = 0;
            for (const v of output) {
                if (Math.abs(v) > outMax) outMax = Math.abs(v);
            }
            outMax = outMax || 1;

            const outNumBars = Math.min(32, output.length);
            const outStep = Math.ceil(output.length / outNumBars);
            const outX = width - padding - maxBarWidth;

            for (let i = 0; i < outNumBars; i++) {
                const idx = i * outStep;
                const value = output[idx];
                const normalized = Math.abs(value) / outMax;
                const barWidth = normalized * maxBarWidth;

                const barY = startY + i * (barHeight + barGap);

                ctx.fillStyle = this.colors.border;
                ctx.fillRect(outX, barY, maxBarWidth, barHeight);

                ctx.fillStyle = value >= 0 ? '#3b82f6' : '#f59e0b';
                ctx.fillRect(outX, barY, barWidth, barHeight);
            }
        }
    }

    /**
     * Render probability distribution bars
     */
    renderProbabilities(canvas, topK, temperature = 1.0) {
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = this.colors.surface;
        ctx.fillRect(0, 0, width, height);

        if (!topK || topK.length === 0) return;

        const padding = 16;
        const titleHeight = 30;
        const labelWidth = 80;
        const percentWidth = 50;

        // Title
        ctx.fillStyle = this.colors.text;
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('Next Token Predictions', padding, padding + 12);

        ctx.fillStyle = this.colors.textMuted;
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText('Temperature: ' + temperature.toFixed(1), width - padding, padding + 12);

        const barHeight = 24;
        const barGap = 8;
        const startY = padding + titleHeight + 10;
        const maxBarWidth = width - padding * 2 - labelWidth - percentWidth;

        // Find max probability
        const maxProb = topK[0]?.probability || 1;

        // Draw bars
        topK.slice(0, 5).forEach((item, i) => {
            const barY = startY + i * (barHeight + barGap);
            const barWidth = (item.probability / maxProb) * maxBarWidth;

            // Token label
            ctx.fillStyle = this.colors.text;
            ctx.font = '11px JetBrains Mono, monospace';
            ctx.textAlign = 'left';
            const displayToken = item.token === ' ' ? '[space]' : item.token;
            ctx.fillText(displayToken.slice(0, 8), padding, barY + barHeight / 2 + 4);

            // Bar background
            ctx.fillStyle = this.colors.border;
            ctx.fillRect(padding + labelWidth, barY, maxBarWidth, barHeight);

            // Probability bar
            const gradient = ctx.createLinearGradient(
                padding + labelWidth, barY,
                padding + labelWidth + barWidth, barY
            );
            gradient.addColorStop(0, '#6366f1');
            gradient.addColorStop(1, '#8b5cf6');
            ctx.fillStyle = gradient;
            ctx.fillRect(padding + labelWidth, barY, barWidth, barHeight);

            // Highlight top prediction
            if (i === 0) {
                ctx.strokeStyle = this.colors.highlight;
                ctx.lineWidth = 2;
                ctx.strokeRect(padding + labelWidth, barY, maxBarWidth, barHeight);
            }

            // Percentage
            ctx.fillStyle = this.colors.text;
            ctx.font = '11px JetBrains Mono, monospace';
            ctx.textAlign = 'right';
            ctx.fillText(item.percentage, width - padding, barY + barHeight / 2 + 4);
        });

        // Formula
        ctx.fillStyle = this.colors.textMuted;
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('P(next) = softmax(logits / T)', width / 2, height - 8);
    }

    /**
     * Render generated tokens sequence
     */
    renderGeneratedTokens(canvas, promptTokens, generatedTokens, currentIdx = -1) {
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = this.colors.surface;
        ctx.fillRect(0, 0, width, height);

        const padding = 12;
        let x = padding;
        const y = height / 2;

        const allTokens = [...promptTokens, ...generatedTokens];

        // Draw tokens
        allTokens.forEach((token, i) => {
            const isPrompt = i < promptTokens.length;
            const isCurrent = i === currentIdx;
            const displayToken = token === ' ' ? '_' : token;

            // Measure text
            ctx.font = '13px JetBrains Mono, monospace';
            const textWidth = ctx.measureText(displayToken).width;
            const boxWidth = textWidth + 12;
            const boxHeight = 28;

            // Box background
            if (isCurrent) {
                ctx.fillStyle = '#fbbf24';
            } else if (isPrompt) {
                ctx.fillStyle = 'rgba(99, 102, 241, 0.3)';
            } else {
                ctx.fillStyle = 'rgba(34, 197, 94, 0.3)';
            }

            this.roundRect(ctx, x, y - boxHeight / 2, boxWidth, boxHeight, 4);
            ctx.fill();

            // Border for current
            if (isCurrent) {
                ctx.strokeStyle = '#f59e0b';
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            // Text
            ctx.fillStyle = isCurrent ? '#0f172a' : this.colors.text;
            ctx.font = '13px JetBrains Mono, monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(displayToken, x + boxWidth / 2, y);

            x += boxWidth + 4;

            // Wrap if needed
            if (x > width - padding - 50) {
                x = padding;
                // Would need multiple lines for long sequences
            }
        });

        // Cursor
        ctx.fillStyle = this.colors.text;
        ctx.fillRect(x, y - 10, 2, 20);
    }

    /**
     * Highlight token connections on attention hover
     */
    drawTokenHighlights(canvas, tokens, queryIdx, keyIdx, weight) {
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;

        // Clear previous highlights
        // This would be called as an overlay

        if (queryIdx === null || keyIdx === null) return;

        // Draw connecting line
        ctx.strokeStyle = this.colors.highlight;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);

        // Positions would depend on layout
        // This is a placeholder for the connection visualization
    }

    // ==================== Utility Functions ====================

    /**
     * Interpolate between two hex colors
     */
    interpolateColor(color1, color2, factor) {
        const c1 = this.hexToRgb(color1);
        const c2 = this.hexToRgb(color2);

        const r = Math.round(c1.r + (c2.r - c1.r) * factor);
        const g = Math.round(c1.g + (c2.g - c1.g) * factor);
        const b = Math.round(c1.b + (c2.b - c1.b) * factor);

        return 'rgb(' + r + ', ' + g + ', ' + b + ')';
    }

    /**
     * Convert hex to RGB
     */
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
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
     * Set hover state for interactive highlights
     */
    setHoverState(headIdx, row, col) {
        this.hoverState = { headIdx, row, col };
    }

    /**
     * Clear hover state
     */
    clearHoverState() {
        this.hoverState = { headIdx: null, row: null, col: null };
    }
}

// Export singleton
export const explainerViz = new ExplainerViz();
