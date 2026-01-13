/**
 * GNN Visualization
 * Canvas-based visualization for Graph Neural Networks
 * Includes force-directed layout, message passing animation, and embedding projection
 */

/**
 * Color scheme
 */
const COLORS = {
    classA: '#3b82f6',       // Blue
    classB: '#ef4444',       // Red
    edge: '#64748b',         // Gray
    edgeLight: '#94a3b8',    // Light gray
    selected: '#fbbf24',     // Yellow
    message: '#22c55e',      // Green
    attention: '#8b5cf6',    // Purple
    background: '#1e293b',   // Dark
    text: '#f8fafc',         // White
    textSecondary: '#94a3b8' // Gray
};

/**
 * GNN Visualizer class
 */
export class GNNVisualizer {
    constructor(graphCanvas, messageCanvas, embeddingCanvas) {
        this.graphCanvas = graphCanvas;
        this.messageCanvas = messageCanvas;
        this.embeddingCanvas = embeddingCanvas;

        this.graphCtx = graphCanvas?.getContext('2d');
        this.messageCtx = messageCanvas?.getContext('2d');
        this.embeddingCtx = embeddingCanvas?.getContext('2d');

        // Graph data
        this.numNodes = 0;
        this.edges = [];
        this.neighbors = [];
        this.labels = [];
        this.predictions = null;

        // Layout state
        this.positions = [];
        this.velocities = [];
        this.layoutIterations = 0;
        this.isLayoutStable = false;

        // Interaction state
        this.selectedNode = null;
        this.hoveredNode = null;

        // Animation state
        this.animationProgress = 0;
        this.currentLayer = 0;
        this.isAnimating = false;
        this.animationFrame = null;

        // Attention weights for GAT
        this.attentionWeights = null;

        // Node radius
        this.nodeRadius = 8;
    }

    /**
     * Initialize with graph data
     */
    setGraph(numNodes, edges, neighbors, labels) {
        this.numNodes = numNodes;
        this.edges = edges;
        this.neighbors = neighbors;
        this.labels = labels;
        this.predictions = null;

        this.initForceLayout();
        this.layoutIterations = 0;
        this.isLayoutStable = false;
    }

    /**
     * Set predictions for coloring
     */
    setPredictions(predictions) {
        this.predictions = predictions;
    }

    /**
     * Set attention weights for GAT visualization
     */
    setAttentionWeights(weights) {
        this.attentionWeights = weights;
    }

    /**
     * Initialize force-directed layout
     */
    initForceLayout() {
        const n = this.numNodes;
        this.positions = [];
        this.velocities = [];

        // Initialize positions in a circle
        const centerX = (this.graphCanvas?.width || 260) / 2;
        const centerY = (this.graphCanvas?.height || 260) / 2;
        const radius = Math.min(centerX, centerY) * 0.7;

        for (let i = 0; i < n; i++) {
            const angle = (2 * Math.PI * i) / n;
            this.positions.push({
                x: centerX + radius * Math.cos(angle) + (Math.random() - 0.5) * 20,
                y: centerY + radius * Math.sin(angle) + (Math.random() - 0.5) * 20
            });
            this.velocities.push({ x: 0, y: 0 });
        }
    }

    /**
     * Update force-directed layout (one iteration)
     */
    updateForceLayout() {
        if (this.isLayoutStable) return;

        const n = this.numNodes;
        const width = this.graphCanvas?.width || 260;
        const height = this.graphCanvas?.height || 260;
        const centerX = width / 2;
        const centerY = height / 2;

        // Force parameters
        const repulsion = 800;
        const attraction = 0.03;
        const damping = 0.85;
        const centerForce = 0.01;

        let totalMovement = 0;

        for (let i = 0; i < n; i++) {
            let fx = 0, fy = 0;

            // Repulsion from all nodes
            for (let j = 0; j < n; j++) {
                if (i === j) continue;

                const dx = this.positions[i].x - this.positions[j].x;
                const dy = this.positions[i].y - this.positions[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy) + 0.1;

                fx += repulsion * dx / (dist * dist);
                fy += repulsion * dy / (dist * dist);
            }

            // Attraction along edges
            for (const j of this.neighbors[i]) {
                const dx = this.positions[j].x - this.positions[i].x;
                const dy = this.positions[j].y - this.positions[i].y;

                fx += attraction * dx;
                fy += attraction * dy;
            }

            // Center gravity
            fx += centerForce * (centerX - this.positions[i].x);
            fy += centerForce * (centerY - this.positions[i].y);

            // Update velocity with damping
            this.velocities[i].x = damping * this.velocities[i].x + fx * 0.1;
            this.velocities[i].y = damping * this.velocities[i].y + fy * 0.1;

            // Limit velocity
            const speed = Math.sqrt(this.velocities[i].x ** 2 + this.velocities[i].y ** 2);
            if (speed > 10) {
                this.velocities[i].x *= 10 / speed;
                this.velocities[i].y *= 10 / speed;
            }

            totalMovement += speed;
        }

        // Update positions
        const padding = 20;
        for (let i = 0; i < n; i++) {
            this.positions[i].x += this.velocities[i].x;
            this.positions[i].y += this.velocities[i].y;

            // Keep within bounds
            this.positions[i].x = Math.max(padding, Math.min(width - padding, this.positions[i].x));
            this.positions[i].y = Math.max(padding, Math.min(height - padding, this.positions[i].y));
        }

        this.layoutIterations++;

        // Check for stability
        if (totalMovement < 0.5 || this.layoutIterations > 300) {
            this.isLayoutStable = true;
        }
    }

    /**
     * Run layout until stable
     */
    runLayoutToStable(maxIterations = 200) {
        for (let i = 0; i < maxIterations && !this.isLayoutStable; i++) {
            this.updateForceLayout();
        }
    }

    /**
     * Get node at position
     */
    getNodeAtPosition(x, y) {
        for (let i = 0; i < this.numNodes; i++) {
            const dx = this.positions[i].x - x;
            const dy = this.positions[i].y - y;
            if (dx * dx + dy * dy < this.nodeRadius * this.nodeRadius * 2) {
                return i;
            }
        }
        return null;
    }

    /**
     * Set selected node
     */
    setSelectedNode(nodeId) {
        this.selectedNode = nodeId;
    }

    /**
     * Render the graph structure
     */
    renderGraph() {
        if (!this.graphCtx) return;

        const ctx = this.graphCtx;
        const width = this.graphCanvas.width;
        const height = this.graphCanvas.height;

        // Clear
        ctx.fillStyle = COLORS.background;
        ctx.fillRect(0, 0, width, height);

        // Draw edges
        ctx.strokeStyle = COLORS.edge;
        ctx.lineWidth = 1;

        for (const [i, j] of this.edges) {
            const p1 = this.positions[i];
            const p2 = this.positions[j];

            if (!p1 || !p2) continue;

            // Draw edge with attention weight if available
            if (this.attentionWeights) {
                const n = this.numNodes;
                const weight = this.attentionWeights[i * n + j] || this.attentionWeights[j * n + i] || 0;
                ctx.strokeStyle = `rgba(139, 92, 246, ${Math.min(weight * 3, 1)})`;
                ctx.lineWidth = 1 + weight * 3;
            } else {
                ctx.strokeStyle = COLORS.edge;
                ctx.lineWidth = 1;
            }

            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }

        // Draw nodes
        for (let i = 0; i < this.numNodes; i++) {
            const pos = this.positions[i];
            if (!pos) continue;

            const label = this.labels[i];
            const predicted = this.predictions ? this.predictions[i] : label;
            const isSelected = this.selectedNode === i;

            // Node color based on true label
            const color = label === 0 ? COLORS.classA : COLORS.classB;

            // Draw selection ring
            if (isSelected) {
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, this.nodeRadius + 4, 0, Math.PI * 2);
                ctx.strokeStyle = COLORS.selected;
                ctx.lineWidth = 3;
                ctx.stroke();
            }

            // Draw prediction ring (if different from label)
            if (this.predictions && predicted !== label) {
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, this.nodeRadius + 2, 0, Math.PI * 2);
                ctx.strokeStyle = predicted === 0 ? COLORS.classA : COLORS.classB;
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            // Draw node
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, this.nodeRadius, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();

            // Draw node ID for small graphs
            if (this.numNodes <= 34) {
                ctx.fillStyle = COLORS.text;
                ctx.font = '8px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(i.toString(), pos.x, pos.y);
            }
        }
    }

    /**
     * Render message passing animation
     */
    renderMessagePassing(layerState, progress = 1) {
        if (!this.messageCtx) return;

        const ctx = this.messageCtx;
        const width = this.messageCanvas.width;
        const height = this.messageCanvas.height;

        // Clear
        ctx.fillStyle = COLORS.background;
        ctx.fillRect(0, 0, width, height);

        // Scale positions to message canvas
        const scaleX = width / (this.graphCanvas?.width || 260);
        const scaleY = height / (this.graphCanvas?.height || 260);

        // Draw edges (faded)
        ctx.strokeStyle = `${COLORS.edge}40`;
        ctx.lineWidth = 1;

        for (const [i, j] of this.edges) {
            const p1 = this.positions[i];
            const p2 = this.positions[j];
            if (!p1 || !p2) continue;

            ctx.beginPath();
            ctx.moveTo(p1.x * scaleX, p1.y * scaleY);
            ctx.lineTo(p2.x * scaleX, p2.y * scaleY);
            ctx.stroke();
        }

        // Draw message arrows
        if (progress > 0) {
            for (const [i, j] of this.edges) {
                this.drawMessageArrow(ctx, i, j, progress, scaleX, scaleY, layerState);
                this.drawMessageArrow(ctx, j, i, progress, scaleX, scaleY, layerState);
            }
        }

        // Draw nodes
        for (let i = 0; i < this.numNodes; i++) {
            const pos = this.positions[i];
            if (!pos) continue;

            const x = pos.x * scaleX;
            const y = pos.y * scaleY;
            const label = this.labels[i];
            const color = label === 0 ? COLORS.classA : COLORS.classB;

            // Glow effect for receiving messages
            if (progress > 0.5) {
                ctx.beginPath();
                ctx.arc(x, y, this.nodeRadius * 1.5 + 4, 0, Math.PI * 2);
                const gradient = ctx.createRadialGradient(x, y, this.nodeRadius, x, y, this.nodeRadius * 1.5 + 4);
                gradient.addColorStop(0, `${COLORS.message}40`);
                gradient.addColorStop(1, 'transparent');
                ctx.fillStyle = gradient;
                ctx.fill();
            }

            ctx.beginPath();
            ctx.arc(x, y, this.nodeRadius * 1.2, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        }

        // Draw title
        ctx.fillStyle = COLORS.text;
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`Layer ${this.currentLayer + 1} Message Passing`, 10, 20);
    }

    /**
     * Draw animated message arrow
     */
    drawMessageArrow(ctx, from, to, progress, scaleX, scaleY, layerState) {
        const p1 = this.positions[from];
        const p2 = this.positions[to];
        if (!p1 || !p2) return;

        const x1 = p1.x * scaleX;
        const y1 = p1.y * scaleY;
        const x2 = p2.x * scaleX;
        const y2 = p2.y * scaleY;

        // Interpolate position based on progress
        const t = Math.min(progress, 1);
        const x = x1 + (x2 - x1) * t;
        const y = y1 + (y2 - y1) * t;

        // Get attention weight if available
        let alpha = 0.6;
        if (layerState?.attentionWeights) {
            const n = this.numNodes;
            const weight = layerState.attentionWeights[to * n + from] || 0;
            alpha = Math.max(0.2, Math.min(weight * 2, 1));
        }

        // Draw arrow head
        const dx = x2 - x1;
        const dy = y2 - y1;
        const angle = Math.atan2(dy, dx);
        const arrowSize = 4;

        ctx.fillStyle = `rgba(34, 197, 94, ${alpha})`;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x - arrowSize * Math.cos(angle - 0.5), y - arrowSize * Math.sin(angle - 0.5));
        ctx.lineTo(x - arrowSize * Math.cos(angle + 0.5), y - arrowSize * Math.sin(angle + 0.5));
        ctx.closePath();
        ctx.fill();
    }

    /**
     * Render node embeddings as 2D scatter plot
     */
    renderEmbeddings(embeddings, numNodes, dim) {
        if (!this.embeddingCtx) return;

        const ctx = this.embeddingCtx;
        const width = this.embeddingCanvas.width;
        const height = this.embeddingCanvas.height;

        // Clear
        ctx.fillStyle = COLORS.background;
        ctx.fillRect(0, 0, width, height);

        if (!embeddings || numNodes === 0) return;

        // Simple 2D projection using first two principal-ish components
        const projected = this.project2D(embeddings, numNodes, dim);

        // Find bounds
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        for (const p of projected) {
            minX = Math.min(minX, p.x);
            maxX = Math.max(maxX, p.x);
            minY = Math.min(minY, p.y);
            maxY = Math.max(maxY, p.y);
        }

        // Add padding
        const rangeX = (maxX - minX) || 1;
        const rangeY = (maxY - minY) || 1;
        const padding = 30;

        // Draw points
        for (let i = 0; i < numNodes; i++) {
            const label = this.labels[i];
            const predicted = this.predictions ? this.predictions[i] : label;
            const color = predicted === 0 ? COLORS.classA : COLORS.classB;

            const x = padding + (projected[i].x - minX) / rangeX * (width - 2 * padding);
            const y = padding + (projected[i].y - minY) / rangeY * (height - 2 * padding);

            // Draw ring if prediction differs from label
            if (this.predictions && predicted !== label) {
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                ctx.strokeStyle = label === 0 ? COLORS.classA : COLORS.classB;
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();

            // Highlight selected
            if (this.selectedNode === i) {
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, Math.PI * 2);
                ctx.strokeStyle = COLORS.selected;
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }

        // Legend
        ctx.font = '10px sans-serif';
        ctx.fillStyle = COLORS.classA;
        ctx.beginPath();
        ctx.arc(width - 80, 15, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = COLORS.text;
        ctx.textAlign = 'left';
        ctx.fillText('Class A', width - 70, 18);

        ctx.fillStyle = COLORS.classB;
        ctx.beginPath();
        ctx.arc(width - 80, 30, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = COLORS.text;
        ctx.fillText('Class B', width - 70, 33);
    }

    /**
     * Simple 2D projection (approximation of PCA)
     */
    project2D(embeddings, numNodes, dim) {
        const projected = [];

        // Use weighted sum of dimensions as simple projection
        // This approximates keeping variance but is much simpler than full PCA
        for (let i = 0; i < numNodes; i++) {
            let x = 0, y = 0;
            for (let j = 0; j < dim; j++) {
                const val = embeddings[i * dim + j];
                // Alternate adding to x and y with different weights
                if (j % 2 === 0) {
                    x += val * Math.cos(j * 0.3);
                } else {
                    y += val * Math.sin(j * 0.3);
                }
            }
            projected.push({ x, y });
        }

        return projected;
    }

    /**
     * Start animation
     */
    startAnimation(onFrame) {
        this.isAnimating = true;
        this.animationProgress = 0;

        const animate = () => {
            if (!this.isAnimating) return;

            this.animationProgress += 0.02;

            if (this.animationProgress >= 1) {
                this.animationProgress = 1;
                this.isAnimating = false;
            }

            if (onFrame) {
                onFrame(this.animationProgress);
            }

            if (this.isAnimating) {
                this.animationFrame = requestAnimationFrame(animate);
            }
        };

        this.animationFrame = requestAnimationFrame(animate);
    }

    /**
     * Stop animation
     */
    stopAnimation() {
        this.isAnimating = false;
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }

    /**
     * Set current layer for visualization
     */
    setCurrentLayer(layer) {
        this.currentLayer = layer;
    }

    /**
     * Clear all canvases
     */
    clear() {
        if (this.graphCtx) {
            this.graphCtx.fillStyle = COLORS.background;
            this.graphCtx.fillRect(0, 0, this.graphCanvas.width, this.graphCanvas.height);
        }
        if (this.messageCtx) {
            this.messageCtx.fillStyle = COLORS.background;
            this.messageCtx.fillRect(0, 0, this.messageCanvas.width, this.messageCanvas.height);
        }
        if (this.embeddingCtx) {
            this.embeddingCtx.fillStyle = COLORS.background;
            this.embeddingCtx.fillRect(0, 0, this.embeddingCanvas.width, this.embeddingCanvas.height);
        }
    }

    /**
     * Resize canvases
     */
    resize() {
        // Re-run layout if needed
        if (this.numNodes > 0 && !this.isLayoutStable) {
            this.runLayoutToStable(50);
        }
    }

    /**
     * Dispose resources
     */
    dispose() {
        this.stopAnimation();
    }
}
