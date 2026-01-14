/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Decision Boundary Visualization - Beautiful Heatmap & Data Points
 * ═══════════════════════════════════════════════════════════════════════════
 */

export class BoundaryVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        this.config = {
            padding: 40,
            resolution: 50,  // Grid resolution for heatmap
            pointRadius: 2.5,
            colors: {
                classA: '#fb7185',     // Rose-pink
                classB: '#38bdf8',     // Sky-cyan
                background: {
                    classA: [251, 113, 133],  // RGB for interpolation (rose)
                    classB: [56, 189, 248]    // RGB for interpolation (cyan)
                },
                grid: 'rgba(148, 163, 184, 0.12)',
                axis: 'rgba(100, 116, 139, 0.25)'
            },
            range: { min: -1.2, max: 1.2 }  // Coordinate range
        };

        this.network = null;
        this.dataset = null;
        this.heatmapData = null;
        this.heatmapImage = null;

        this.init();
    }

    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();

        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);

        this.width = rect.width;
        this.height = rect.height;

        // Recalculate heatmap if network exists
        if (this.network) {
            this.updateHeatmap();
        }
    }

    /**
     * Set the neural network for predictions
     */
    setNetwork(network) {
        this.network = network;
    }

    /**
     * Set dataset for visualization
     */
    setDataset(dataset) {
        this.dataset = dataset;
    }

    /**
     * Convert data coordinates to canvas coordinates
     */
    dataToCanvas(x, y) {
        const { min, max } = this.config.range;
        const range = max - min;

        const canvasX = this.config.padding +
            ((x - min) / range) * (this.width - this.config.padding * 2);
        const canvasY = this.config.padding +
            ((max - y) / range) * (this.height - this.config.padding * 2);

        return { x: canvasX, y: canvasY };
    }

    /**
     * Convert canvas coordinates to data coordinates
     */
    canvasToData(canvasX, canvasY) {
        const { min, max } = this.config.range;
        const range = max - min;

        const x = min + ((canvasX - this.config.padding) /
            (this.width - this.config.padding * 2)) * range;
        const y = max - ((canvasY - this.config.padding) /
            (this.height - this.config.padding * 2)) * range;

        return { x, y };
    }

    /**
     * Update the decision boundary heatmap
     */
    updateHeatmap() {
        if (!this.network) return;

        const { min, max } = this.config.range;
        const resolution = this.config.resolution;
        const step = (max - min) / resolution;

        // Create grid of points
        const points = [];
        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const x = min + j * step + step / 2;
                const y = max - i * step - step / 2;
                points.push([x, y]);
            }
        }

        // Get predictions for all grid points
        const { Tensor } = this.network.layers[0].input ?
            { Tensor: this.network.layers[0].input.constructor } :
            { Tensor: null };

        if (!Tensor) {
            // Import dynamically if needed
            this.heatmapData = new Array(resolution * resolution).fill(0.5);
            return;
        }

        const inputTensor = new Tensor(points.length, 2, new Float32Array(points.flat()));
        const predictions = this.network.predict(inputTensor);

        this.heatmapData = Array.from(predictions.data);
        this.renderHeatmapToImage();
    }

    /**
     * Render heatmap to an offscreen canvas for performance
     */
    renderHeatmapToImage() {
        if (!this.heatmapData) return;

        const resolution = this.config.resolution;
        const offscreen = document.createElement('canvas');
        offscreen.width = resolution;
        offscreen.height = resolution;
        const offCtx = offscreen.getContext('2d');

        const imageData = offCtx.createImageData(resolution, resolution);
        const data = imageData.data;

        const { classA, classB } = this.config.colors.background;

        for (let i = 0; i < this.heatmapData.length; i++) {
            const prediction = this.heatmapData[i];
            const idx = i * 4;

            // Interpolate between class colors
            const t = prediction;
            data[idx] = Math.round(classB[0] * (1 - t) + classA[0] * t);
            data[idx + 1] = Math.round(classB[1] * (1 - t) + classA[1] * t);
            data[idx + 2] = Math.round(classB[2] * (1 - t) + classA[2] * t);
            data[idx + 3] = 180; // Alpha
        }

        offCtx.putImageData(imageData, 0, 0);
        this.heatmapImage = offscreen;
    }

    /**
     * Draw coordinate grid
     */
    drawGrid() {
        const { min, max } = this.config.range;
        const numLines = 5;
        const step = (max - min) / numLines;

        this.ctx.strokeStyle = this.config.colors.grid;
        this.ctx.lineWidth = 1;

        // Vertical lines
        for (let i = 0; i <= numLines; i++) {
            const x = min + i * step;
            const start = this.dataToCanvas(x, min);
            const end = this.dataToCanvas(x, max);

            this.ctx.beginPath();
            this.ctx.moveTo(start.x, start.y);
            this.ctx.lineTo(end.x, end.y);
            this.ctx.stroke();
        }

        // Horizontal lines
        for (let i = 0; i <= numLines; i++) {
            const y = min + i * step;
            const start = this.dataToCanvas(min, y);
            const end = this.dataToCanvas(max, y);

            this.ctx.beginPath();
            this.ctx.moveTo(start.x, start.y);
            this.ctx.lineTo(end.x, end.y);
            this.ctx.stroke();
        }

        // Axes
        this.ctx.strokeStyle = this.config.colors.axis;
        this.ctx.lineWidth = 2;

        // X axis
        let start = this.dataToCanvas(min, 0);
        let end = this.dataToCanvas(max, 0);
        this.ctx.beginPath();
        this.ctx.moveTo(start.x, start.y);
        this.ctx.lineTo(end.x, end.y);
        this.ctx.stroke();

        // Y axis
        start = this.dataToCanvas(0, min);
        end = this.dataToCanvas(0, max);
        this.ctx.beginPath();
        this.ctx.moveTo(start.x, start.y);
        this.ctx.lineTo(end.x, end.y);
        this.ctx.stroke();

        // Axis labels
        this.ctx.fillStyle = '#475569';
        this.ctx.font = '11px Inter, sans-serif';
        this.ctx.textAlign = 'center';

        // X labels
        for (let i = 0; i <= numLines; i++) {
            const val = min + i * step;
            const pos = this.dataToCanvas(val, min);
            this.ctx.fillText(val.toFixed(1), pos.x, pos.y + 15);
        }

        // Y labels
        this.ctx.textAlign = 'right';
        for (let i = 0; i <= numLines; i++) {
            const val = min + i * step;
            const pos = this.dataToCanvas(min, val);
            this.ctx.fillText(val.toFixed(1), pos.x - 8, pos.y + 4);
        }
    }

    /**
     * Draw heatmap background
     */
    drawHeatmap() {
        if (!this.heatmapImage) return;

        const drawWidth = this.width - this.config.padding * 2;
        const drawHeight = this.height - this.config.padding * 2;

        // Enable image smoothing for better quality
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';

        // Draw scaled heatmap
        this.ctx.drawImage(
            this.heatmapImage,
            this.config.padding,
            this.config.padding,
            drawWidth,
            drawHeight
        );
    }

    /**
     * Draw decision boundary contour (50% line)
     */
    drawContour() {
        if (!this.heatmapData) return;

        const resolution = this.config.resolution;
        const { min, max } = this.config.range;
        const step = (max - min) / resolution;

        this.ctx.strokeStyle = 'rgba(51, 65, 85, 0.5)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);

        // Simple marching squares-ish contour
        for (let i = 0; i < resolution - 1; i++) {
            for (let j = 0; j < resolution - 1; j++) {
                const idx = i * resolution + j;
                const v00 = this.heatmapData[idx];
                const v10 = this.heatmapData[idx + 1];
                const v01 = this.heatmapData[idx + resolution];
                const v11 = this.heatmapData[idx + resolution + 1];

                // Check if contour passes through this cell
                const threshold = 0.5;
                const above = [v00 > threshold, v10 > threshold, v01 > threshold, v11 > threshold];
                const numAbove = above.filter(Boolean).length;

                if (numAbove > 0 && numAbove < 4) {
                    // Contour passes through this cell
                    const x = min + j * step + step / 2;
                    const y = max - i * step - step / 2;
                    const pos = this.dataToCanvas(x, y);

                    // Draw a small segment
                    this.ctx.beginPath();
                    this.ctx.arc(pos.x, pos.y, 2, 0, Math.PI * 2);
                    this.ctx.stroke();
                }
            }
        }

        this.ctx.setLineDash([]);
    }

    /**
     * Draw data points
     */
    drawDataPoints() {
        if (!this.dataset) return;

        const { X, y } = this.dataset;

        for (let i = 0; i < X.rows; i++) {
            const x1 = X.get(i, 0);
            const x2 = X.get(i, 1);
            const label = y.get(i, 0);

            const pos = this.dataToCanvas(x1, x2);
            const color = label >= 0.5 ? this.config.colors.classA : this.config.colors.classB;
            const radius = this.config.pointRadius;

            // Outer glow
            const gradient = this.ctx.createRadialGradient(
                pos.x, pos.y, 0,
                pos.x, pos.y, radius * 2
            );
            gradient.addColorStop(0, color + '80');
            gradient.addColorStop(1, 'transparent');

            this.ctx.beginPath();
            this.ctx.arc(pos.x, pos.y, radius * 2, 0, Math.PI * 2);
            this.ctx.fillStyle = gradient;
            this.ctx.fill();

            // Inner circle
            this.ctx.beginPath();
            this.ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            this.ctx.fillStyle = color;
            this.ctx.fill();

            // Dark border for light theme
            this.ctx.strokeStyle = 'rgba(51, 65, 85, 0.6)';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        }
    }

    /**
     * Main render function
     */
    render() {
        // Clear
        this.ctx.clearRect(0, 0, this.width, this.height);

        // Draw layers
        this.drawHeatmap();
        this.drawGrid();
        this.drawDataPoints();
    }

    /**
     * Quick update (just data points, no heatmap recalc)
     */
    quickRender() {
        this.ctx.clearRect(0, 0, this.width, this.height);
        this.drawHeatmap();
        this.drawGrid();
        this.drawDataPoints();
    }
}
