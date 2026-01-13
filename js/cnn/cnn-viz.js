/**
 * CNN Visualization
 * Renders CNN architecture diagrams and feature map visualizations
 */

import { LAYER_TYPES } from './cnn-model.js';

export class CNNVisualizer {
    constructor(archCanvas, featureCanvas) {
        this.archCanvas = archCanvas;
        this.featureCanvas = featureCanvas;
        this.archCtx = archCanvas?.getContext('2d');
        this.featureCtx = featureCanvas?.getContext('2d');

        this.model = null;
        this.layerConfigs = [];
        this.layerShapes = [];  // Output shapes for each layer
        this.selectedLayerIndex = 0;

        // Layout settings
        this.archPadding = 50;
        this.layerSpacing = 90;
        this.maxBlockHeight = 120;
        this.maxBlockWidth = 45;

        // Enhanced visual settings
        this.perspective = 0.4;  // 3D perspective factor
        this.sliceCount = 5;     // Number of feature map slices to show
        this.sliceOffset = 4;    // Offset between slices
        this.shadowBlur = 20;
        this.connectionCurve = 0.3;  // Bezier curve tension

        // Animation
        this.flowParticles = [];
        this.flowSpeed = 2;

        // Feature map settings
        this.featureMapSize = 40;  // Size of each feature map tile
        this.featureMapPadding = 4;
        this.featureMapCols = 8;   // Feature maps per row

        // Animation state
        this.animationFrame = null;
        this.pulsePhase = 0;
    }

    /**
     * Set model and layer configurations for visualization
     * @param {tf.LayersModel} model - Built TensorFlow.js model
     * @param {Array} layerConfigs - Array of layer config objects
     */
    setModel(model, layerConfigs) {
        this.model = model;
        this.layerConfigs = layerConfigs;
        this.layerShapes = [];

        if (model && model.layers) {
            for (const layer of model.layers) {
                this.layerShapes.push(layer.outputShape);
            }
        }

        this.renderArchitecture();
    }

    /**
     * Set selected layer for feature map display
     * @param {number} index - Layer index
     */
    setSelectedLayer(index) {
        if (index >= 0 && index < this.layerConfigs.length) {
            this.selectedLayerIndex = index;
            this.renderArchitecture();
        }
    }

    /**
     * Get layer info at canvas position
     * @param {number} x - Canvas X coordinate
     * @param {number} y - Canvas Y coordinate
     * @returns {number|null} Layer index or null
     */
    getLayerAtPosition(x, y) {
        if (!this.layerBounds) return null;

        for (let i = 0; i < this.layerBounds.length; i++) {
            const bounds = this.layerBounds[i];
            if (x >= bounds.x && x <= bounds.x + bounds.width &&
                y >= bounds.y && y <= bounds.y + bounds.height) {
                return i;
            }
        }
        return null;
    }

    /**
     * Render the CNN architecture diagram
     */
    renderArchitecture() {
        if (!this.archCtx || !this.layerConfigs.length) return;

        const canvas = this.archCanvas;
        const ctx = this.archCtx;
        const dpr = window.devicePixelRatio || 1;
        const width = canvas.width / dpr;
        const height = canvas.height / dpr;

        // Clear canvas with subtle gradient background
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const bgGradient = ctx.createLinearGradient(0, 0, 0, height);
        bgGradient.addColorStop(0, '#0f172a');
        bgGradient.addColorStop(1, '#1e293b');
        ctx.fillStyle = bgGradient;
        ctx.fillRect(0, 0, width, height);

        // Draw subtle grid pattern
        this.drawBackgroundGrid(ctx, width, height);

        // Calculate layout
        const totalLayers = this.layerConfigs.length + 1;  // +1 for input
        const availableWidth = width - this.archPadding * 2;
        const spacing = Math.min(this.layerSpacing, availableWidth / totalLayers);
        const startX = this.archPadding + 20;
        const centerY = height / 2;

        this.layerBounds = [];
        const layerPositions = [];

        // Calculate input layer info
        const inputShape = this.layerShapes[0] ?
            [1, ...this.layerShapes[0].slice(1)] : [1, 28, 28, 1];

        // First pass: calculate all layer positions
        let x = startX;
        layerPositions.push({ x, type: 'input', shape: inputShape, config: null, index: -1 });
        x += spacing;

        for (let i = 0; i < this.layerConfigs.length; i++) {
            const config = this.layerConfigs[i];
            const shape = this.layerShapes[i] || [1, 14, 14, 32];
            layerPositions.push({ x, type: config.type, shape, config: config.config, index: i });
            x += spacing;
        }

        // Draw connections first (behind layers)
        for (let i = 1; i < layerPositions.length; i++) {
            const prev = layerPositions[i - 1];
            const curr = layerPositions[i];
            this.drawConnection(ctx, prev, curr, centerY, spacing);
        }

        // Draw each layer
        for (let i = 0; i < layerPositions.length; i++) {
            const pos = layerPositions[i];
            const bounds = this.drawLayer(ctx, pos.x, centerY, {
                type: pos.type,
                config: pos.config,
                shape: pos.shape
            }, pos.index);

            if (pos.index >= 0) {
                this.layerBounds.push(bounds);
            }
        }

        // Draw title
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 14px system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('Network Architecture', 16, 24);

        // Draw compact model info
        this.drawModelInfo(ctx, width, height);
    }

    /**
     * Draw subtle background grid
     */
    drawBackgroundGrid(ctx, width, height) {
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.05)';
        ctx.lineWidth = 1;

        const gridSize = 30;
        for (let x = 0; x < width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = 0; y < height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
    }

    /**
     * Draw connection between layers with gradient and curve
     */
    drawConnection(ctx, prev, curr, centerY, spacing) {
        const prevLayerType = LAYER_TYPES[prev.type];
        const currLayerType = LAYER_TYPES[curr.type];
        const prevColor = prev.type === 'input' ? '#64748b' : (prevLayerType?.color || '#64748b');
        const currColor = curr.type === 'input' ? '#64748b' : (currLayerType?.color || '#64748b');

        // Calculate connection points based on layer dimensions
        const prevDims = this.getLayerDimensions(prev.shape, prev.type);
        const currDims = this.getLayerDimensions(curr.shape, curr.type);

        const x1 = prev.x + prevDims.totalWidth;
        const x2 = curr.x;
        const y1 = centerY;
        const y2 = centerY;

        // Create gradient for connection
        const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
        gradient.addColorStop(0, this.hexToRgba(prevColor, 0.6));
        gradient.addColorStop(0.5, 'rgba(148, 163, 184, 0.3)');
        gradient.addColorStop(1, this.hexToRgba(currColor, 0.6));

        // Draw curved connection with multiple lines for depth
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 2;

        // Draw main bezier curve
        const cpOffset = (x2 - x1) * this.connectionCurve;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.bezierCurveTo(x1 + cpOffset, y1, x2 - cpOffset, y2, x2, y2);
        ctx.stroke();

        // Draw arrow at end
        this.drawArrow(ctx, x2 - 8, y2, currColor);
    }

    /**
     * Draw arrow indicator
     */
    drawArrow(ctx, x, y, color) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(x, y - 4);
        ctx.lineTo(x + 8, y);
        ctx.lineTo(x, y + 4);
        ctx.closePath();
        ctx.fill();
    }

    /**
     * Get layer visual dimensions
     */
    getLayerDimensions(shape, type) {
        let blockWidth = 40;
        let blockHeight = 100;
        let depth = 20;
        let slices = 1;

        if (shape && shape.length >= 3) {
            // Conv/Pool layers: [batch, height, width, channels]
            const [_, h, w, c] = shape.length === 4 ? shape : [1, shape[1], shape[2], shape[3] || 1];
            blockHeight = Math.min(this.maxBlockHeight, Math.max(50, (h || 28) * 3));
            blockWidth = Math.min(this.maxBlockWidth, Math.max(25, (w || 28) * 1.5));
            slices = Math.min(this.sliceCount, Math.max(2, Math.ceil((c || 1) / 16)));
            depth = slices * this.sliceOffset + 15;
        } else if (shape && shape.length === 2) {
            // Dense layers: [batch, units]
            const units = shape[1] || 128;
            blockHeight = Math.min(this.maxBlockHeight, Math.max(60, Math.sqrt(units) * 8));
            blockWidth = 25;
            depth = 10;
            slices = 1;
        }

        // Special handling for flatten/dropout - make them thinner
        if (type === 'flatten' || type === 'dropout' || type === 'batchNormalization') {
            blockWidth = 15;
            depth = 8;
        }

        return { blockWidth, blockHeight, depth, slices, totalWidth: blockWidth + depth };
    }

    /**
     * Draw a single layer block with enhanced 3D visualization
     */
    drawLayer(ctx, x, centerY, layerInfo, index) {
        const { type, config, shape } = layerInfo;
        const isSelected = index === this.selectedLayerIndex;

        // Get layer visual properties
        const layerType = LAYER_TYPES[type];
        const baseColor = type === 'input' ? '#64748b' : (layerType?.color || '#64748b');

        // Get dimensions
        const dims = this.getLayerDimensions(shape, type);
        const { blockWidth, blockHeight, depth, slices } = dims;
        const topY = centerY - blockHeight / 2;

        // Different rendering based on layer type
        if (type === 'conv2d' || type === 'maxPooling2d' || type === 'avgPooling2d' || type === 'input') {
            // Draw stacked feature maps for conv/pool/input layers
            this.drawStackedFeatureMaps(ctx, x, topY, blockWidth, blockHeight, slices, baseColor, isSelected);
        } else if (type === 'dense') {
            // Draw neurons column for dense layers
            this.drawDenseNeurons(ctx, x, topY, blockWidth, blockHeight, shape, baseColor, isSelected);
        } else if (type === 'flatten') {
            // Draw flatten transition
            this.drawFlattenLayer(ctx, x, topY, blockWidth, blockHeight, baseColor, isSelected);
        } else if (type === 'dropout') {
            // Draw dropout pattern
            this.drawDropoutLayer(ctx, x, topY, blockWidth, blockHeight, baseColor, isSelected, config?.rate || 0.25);
        } else if (type === 'batchNormalization') {
            // Draw batch norm indicator
            this.drawBatchNormLayer(ctx, x, topY, blockWidth, blockHeight, baseColor, isSelected);
        } else {
            // Default block rendering
            this.drawBasicBlock(ctx, x, topY, blockWidth, blockHeight, depth, baseColor, isSelected);
        }

        // Draw layer label below
        const labelY = topY + blockHeight + 14;
        const centerX = x + blockWidth / 2 + (type === 'dense' || type === 'flatten' || type === 'dropout' || type === 'batchNormalization' ? 0 : depth / 2);

        // Layer name - use shorter names to avoid overlap
        ctx.fillStyle = isSelected ? '#f8fafc' : '#cbd5e1';
        ctx.font = isSelected ? 'bold 10px system-ui, sans-serif' : '10px system-ui, sans-serif';
        ctx.textAlign = 'center';
        let name = type === 'input' ? 'Input' : (layerType?.name || type);
        // Shorten some names
        if (name === 'MaxPool2D') name = 'MaxPool';
        if (name === 'AvgPool2D') name = 'AvgPool';
        if (name === 'BatchNorm') name = 'BN';
        ctx.fillText(name, centerX, labelY);

        // Shape info
        ctx.fillStyle = '#64748b';
        ctx.font = '8px "JetBrains Mono", monospace';
        if (shape) {
            const shapeStr = this.formatShape(shape);
            ctx.fillText(shapeStr, centerX, labelY + 11);
        }

        // Config badge above layer
        if (config) {
            this.drawConfigBadge(ctx, centerX, topY - 10, type, config, baseColor);
        }

        return { x, y: topY, width: blockWidth + depth, height: blockHeight };
    }

    /**
     * Draw stacked feature map slices for conv/pool layers
     */
    drawStackedFeatureMaps(ctx, x, topY, width, height, slices, color, isSelected) {
        const offset = this.sliceOffset;

        // Draw slices from back to front
        for (let i = slices - 1; i >= 0; i--) {
            const sliceX = x + i * offset;
            const sliceY = topY - i * offset;

            // Create gradient for each slice
            const gradient = ctx.createLinearGradient(sliceX, sliceY, sliceX + width, sliceY + height);
            const lightColor = this.lightenColor(color, 0.2 - i * 0.05);
            const darkColor = this.darkenColor(color, 0.1 + i * 0.05);
            gradient.addColorStop(0, lightColor);
            gradient.addColorStop(1, darkColor);

            // Draw shadow for depth
            if (i === 0) {
                ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
                ctx.shadowBlur = 10;
                ctx.shadowOffsetX = 5;
                ctx.shadowOffsetY = 5;
            }

            // Draw slice rectangle with rounded corners
            ctx.fillStyle = gradient;
            this.roundRect(ctx, sliceX, sliceY, width, height, 4);
            ctx.fill();

            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 0;

            // Draw border
            ctx.strokeStyle = isSelected && i === 0 ? '#fff' : this.darkenColor(color, 0.3);
            ctx.lineWidth = isSelected && i === 0 ? 2 : 1;
            this.roundRect(ctx, sliceX, sliceY, width, height, 4);
            ctx.stroke();

            // Draw subtle inner pattern on front slice
            if (i === 0) {
                this.drawFeatureMapPattern(ctx, sliceX, sliceY, width, height, color);
            }
        }

        // Selection glow
        if (isSelected) {
            ctx.shadowColor = color;
            ctx.shadowBlur = 20;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            this.roundRect(ctx, x, topY, width, height, 4);
            ctx.stroke();
            ctx.shadowBlur = 0;
        }
    }

    /**
     * Draw subtle pattern inside feature map to suggest activation
     */
    drawFeatureMapPattern(ctx, x, y, width, height, color) {
        ctx.save();
        ctx.globalAlpha = 0.15;

        const gridSize = Math.min(width, height) / 4;
        for (let gx = 0; gx < 4; gx++) {
            for (let gy = 0; gy < 4; gy++) {
                // Random-ish pattern based on position
                if ((gx + gy) % 2 === 0) {
                    const intensity = 0.3 + Math.sin(gx * gy) * 0.2;
                    ctx.fillStyle = this.lightenColor(color, intensity);
                    ctx.fillRect(
                        x + 4 + gx * (width - 8) / 4,
                        y + 4 + gy * (height - 8) / 4,
                        (width - 8) / 4 - 2,
                        (height - 8) / 4 - 2
                    );
                }
            }
        }

        ctx.restore();
    }

    /**
     * Draw dense layer as column of neuron circles
     */
    drawDenseNeurons(ctx, x, topY, width, height, shape, color, isSelected) {
        const units = shape && shape[1] ? shape[1] : 128;
        const maxNeurons = 8;  // Maximum neurons to display
        const neuronCount = Math.min(maxNeurons, units);
        const neuronRadius = Math.min(10, (height - 20) / (neuronCount * 2.5));
        const spacing = (height - 20) / (neuronCount + 1);
        const centerX = x + width / 2;

        // Draw background pill shape
        ctx.fillStyle = 'rgba(15, 23, 42, 0.5)';
        this.roundRect(ctx, x - 2, topY, width + 4, height, width / 2);
        ctx.fill();

        // Selection glow
        if (isSelected) {
            ctx.shadowColor = color;
            ctx.shadowBlur = 15;
        }

        // Draw neurons
        for (let i = 0; i < neuronCount; i++) {
            const neuronY = topY + 10 + spacing * (i + 1);

            // Neuron gradient
            const gradient = ctx.createRadialGradient(
                centerX - 2, neuronY - 2, 0,
                centerX, neuronY, neuronRadius
            );
            gradient.addColorStop(0, this.lightenColor(color, 0.4));
            gradient.addColorStop(0.7, color);
            gradient.addColorStop(1, this.darkenColor(color, 0.3));

            ctx.beginPath();
            ctx.arc(centerX, neuronY, neuronRadius, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();

            ctx.strokeStyle = this.darkenColor(color, 0.2);
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        ctx.shadowBlur = 0;

        // Draw ellipsis if more neurons
        if (units > maxNeurons) {
            ctx.fillStyle = '#94a3b8';
            ctx.font = 'bold 14px system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('···', centerX, topY + height - 5);
        }

        // Draw border
        ctx.strokeStyle = isSelected ? '#fff' : this.darkenColor(color, 0.3);
        ctx.lineWidth = isSelected ? 2 : 1;
        this.roundRect(ctx, x - 2, topY, width + 4, height, width / 2);
        ctx.stroke();
    }

    /**
     * Draw flatten layer as transition indicator
     */
    drawFlattenLayer(ctx, x, topY, width, height, color, isSelected) {
        const centerX = x + width / 2;
        const centerY = topY + height / 2;

        // Draw converging lines
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;

        const lines = 5;
        for (let i = 0; i < lines; i++) {
            const startY = topY + (height / (lines + 1)) * (i + 1);
            const startX = x - 10;
            const endX = x + width + 10;

            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.lineTo(centerX, centerY);
            ctx.lineTo(endX, centerY);
            ctx.stroke();
        }

        // Center dot
        ctx.beginPath();
        ctx.arc(centerX, centerY, 6, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        if (isSelected) {
            ctx.shadowColor = color;
            ctx.shadowBlur = 10;
            ctx.stroke();
            ctx.shadowBlur = 0;
        }
    }

    /**
     * Draw dropout layer with dropout pattern
     */
    drawDropoutLayer(ctx, x, topY, width, height, color, isSelected, rate) {
        const centerX = x + width / 2;

        // Background
        ctx.fillStyle = 'rgba(15, 23, 42, 0.5)';
        this.roundRect(ctx, x - 2, topY, width + 4, height, 4);
        ctx.fill();

        // Draw dropout pattern (some dots filled, some empty)
        const rows = 6;
        const dotRadius = 4;
        const spacing = (height - 20) / rows;

        for (let i = 0; i < rows; i++) {
            const dotY = topY + 10 + spacing * (i + 0.5);
            const isDropped = Math.random() < rate;

            ctx.beginPath();
            ctx.arc(centerX, dotY, dotRadius, 0, Math.PI * 2);

            if (isDropped) {
                // Empty (dropped) circle
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5;
                ctx.setLineDash([2, 2]);
                ctx.stroke();
                ctx.setLineDash([]);
            } else {
                // Filled circle
                ctx.fillStyle = color;
                ctx.fill();
            }
        }

        // Border
        ctx.strokeStyle = isSelected ? '#fff' : this.darkenColor(color, 0.3);
        ctx.lineWidth = isSelected ? 2 : 1;
        this.roundRect(ctx, x - 2, topY, width + 4, height, 4);
        ctx.stroke();
    }

    /**
     * Draw batch normalization layer
     */
    drawBatchNormLayer(ctx, x, topY, width, height, color, isSelected) {
        const centerX = x + width / 2;

        // Background
        ctx.fillStyle = 'rgba(15, 23, 42, 0.5)';
        this.roundRect(ctx, x - 2, topY, width + 4, height, 4);
        ctx.fill();

        // Draw wave pattern to indicate normalization
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        const waveHeight = 15;
        const waveFreq = 3;
        for (let py = topY + 15; py < topY + height - 15; py += 2) {
            const offset = Math.sin((py - topY) * 0.15 * waveFreq) * (waveHeight / 2);
            if (py === topY + 15) {
                ctx.moveTo(centerX + offset, py);
            } else {
                ctx.lineTo(centerX + offset, py);
            }
        }
        ctx.stroke();

        // Draw mean/std indicators
        ctx.fillStyle = color;
        ctx.font = '8px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('μ', centerX - 8, topY + height / 2 - 5);
        ctx.fillText('σ', centerX + 8, topY + height / 2 + 10);

        // Border
        ctx.strokeStyle = isSelected ? '#fff' : this.darkenColor(color, 0.3);
        ctx.lineWidth = isSelected ? 2 : 1;
        this.roundRect(ctx, x - 2, topY, width + 4, height, 4);
        ctx.stroke();
    }

    /**
     * Draw basic 3D block (fallback)
     */
    drawBasicBlock(ctx, x, topY, width, height, depth, color, isSelected) {
        // 3D effect - side
        ctx.fillStyle = this.darkenColor(color, 0.3);
        ctx.beginPath();
        ctx.moveTo(x + width, topY);
        ctx.lineTo(x + width + depth, topY - depth * 0.6);
        ctx.lineTo(x + width + depth, topY - depth * 0.6 + height);
        ctx.lineTo(x + width, topY + height);
        ctx.closePath();
        ctx.fill();

        // 3D effect - top
        ctx.fillStyle = this.darkenColor(color, 0.15);
        ctx.beginPath();
        ctx.moveTo(x, topY);
        ctx.lineTo(x + depth, topY - depth * 0.6);
        ctx.lineTo(x + width + depth, topY - depth * 0.6);
        ctx.lineTo(x + width, topY);
        ctx.closePath();
        ctx.fill();

        // Main face
        const gradient = ctx.createLinearGradient(x, topY, x + width, topY + height);
        gradient.addColorStop(0, this.lightenColor(color, 0.1));
        gradient.addColorStop(1, this.darkenColor(color, 0.1));
        ctx.fillStyle = gradient;

        if (isSelected) {
            ctx.shadowColor = color;
            ctx.shadowBlur = 15;
        }
        ctx.fillRect(x, topY, width, height);
        ctx.shadowBlur = 0;

        // Border
        ctx.strokeStyle = isSelected ? '#fff' : this.darkenColor(color, 0.2);
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.strokeRect(x, topY, width, height);
    }

    /**
     * Draw config badge above layer
     */
    drawConfigBadge(ctx, centerX, y, type, config, color) {
        let text = '';
        if (type === 'conv2d') {
            text = `${config.filters}@${config.kernelSize}×${config.kernelSize}`;
        } else if (type === 'dense') {
            text = `${config.units}`;
        } else if (type === 'dropout') {
            text = `${(config.rate * 100).toFixed(0)}%`;
        } else if (type === 'maxPooling2d' || type === 'avgPooling2d') {
            text = `${config.poolSize}×${config.poolSize}`;
        }

        if (!text) return;

        // Measure text
        ctx.font = '9px "JetBrains Mono", monospace';
        const metrics = ctx.measureText(text);
        const padding = 6;
        const badgeWidth = metrics.width + padding * 2;
        const badgeHeight = 16;
        const badgeX = centerX - badgeWidth / 2;
        const badgeY = y - badgeHeight;

        // Draw badge background
        ctx.fillStyle = this.hexToRgba(color, 0.2);
        this.roundRect(ctx, badgeX, badgeY, badgeWidth, badgeHeight, 3);
        ctx.fill();

        // Draw badge border
        ctx.strokeStyle = this.hexToRgba(color, 0.5);
        ctx.lineWidth = 1;
        this.roundRect(ctx, badgeX, badgeY, badgeWidth, badgeHeight, 3);
        ctx.stroke();

        // Draw text
        ctx.fillStyle = this.lightenColor(color, 0.3);
        ctx.textAlign = 'center';
        ctx.fillText(text, centerX, badgeY + 11);
    }

    /**
     * Draw model info summary
     */
    drawModelInfo(ctx, width, height) {
        if (!this.model) return;

        const totalLayers = this.layerConfigs.length;
        let totalParams = 0;
        if (this.model.layers) {
            for (const layer of this.model.layers) {
                const weights = layer.getWeights();
                weights.forEach(w => { totalParams += w.size; });
            }
        }

        const text = `${totalLayers} layers · ${this.formatNumber(totalParams)} params`;
        ctx.fillStyle = '#64748b';
        ctx.font = '10px system-ui, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(text, width - 16, height - 12);
    }

    /**
     * Format large numbers with K/M suffix
     */
    formatNumber(num) {
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return num.toString();
    }

    /**
     * Helper: Draw rounded rectangle path
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
     * Helper: Convert hex to rgba
     */
    hexToRgba(hex, alpha) {
        const num = parseInt(hex.replace('#', ''), 16);
        const r = (num >> 16) & 255;
        const g = (num >> 8) & 255;
        const b = num & 255;
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    /**
     * Helper: Lighten a hex color
     */
    lightenColor(hex, amount) {
        const num = parseInt(hex.replace('#', ''), 16);
        const r = Math.min(255, (num >> 16) + Math.floor(255 * amount));
        const g = Math.min(255, ((num >> 8) & 0x00FF) + Math.floor(255 * amount));
        const b = Math.min(255, (num & 0x0000FF) + Math.floor(255 * amount));
        return `rgb(${r}, ${g}, ${b})`;
    }

    /**
     * Draw legend (simplified - visual design is now self-explanatory)
     */
    drawLegend(ctx, x, y) {
        // Legend is no longer needed as each layer type has
        // distinct visual representation (stacked maps, neurons, etc.)
    }

    /**
     * Format shape array for display
     */
    formatShape(shape) {
        if (!shape || shape.length === 0) return '';
        // Skip batch dimension
        const dims = shape.slice(1).filter(d => d !== null);
        return dims.join('x');
    }

    /**
     * Darken a hex color
     */
    darkenColor(hex, amount) {
        const num = parseInt(hex.replace('#', ''), 16);
        const r = Math.max(0, (num >> 16) - Math.floor(255 * amount));
        const g = Math.max(0, ((num >> 8) & 0x00FF) - Math.floor(255 * amount));
        const b = Math.max(0, (num & 0x0000FF) - Math.floor(255 * amount));
        return `rgb(${r}, ${g}, ${b})`;
    }

    /**
     * Render feature maps for a specific layer
     * @param {tf.Tensor} activations - Activation tensor for the layer
     * @param {number} layerIndex - Layer index
     */
    async renderFeatureMaps(activations, layerIndex = this.selectedLayerIndex) {
        if (!this.featureCtx || !activations) return;

        const canvas = this.featureCanvas;
        const ctx = this.featureCtx;
        const dpr = window.devicePixelRatio || 1;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const shape = activations.shape;

        // Handle different activation shapes
        if (shape.length === 4) {
            // Conv/Pool output: [batch, height, width, channels]
            await this.renderConvFeatureMaps(ctx, activations, canvas.width / dpr, canvas.height / dpr);
        } else if (shape.length === 2) {
            // Dense output: [batch, units]
            await this.renderDenseActivations(ctx, activations, canvas.width / dpr, canvas.height / dpr);
        }

        // Draw layer info
        ctx.fillStyle = '#f8fafc';
        ctx.font = '12px system-ui, sans-serif';
        ctx.textAlign = 'left';
        const layerName = this.layerConfigs[layerIndex]?.type || 'Layer';
        ctx.fillText(`Layer ${layerIndex + 1}: ${LAYER_TYPES[layerName]?.name || layerName}`, 10, 20);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px monospace';
        ctx.fillText(`Shape: ${this.formatShape(shape)}`, 10, 35);
    }

    /**
     * Render convolutional layer feature maps
     */
    async renderConvFeatureMaps(ctx, activations, width, height) {
        const [batch, h, w, channels] = activations.shape;
        const data = await activations.data();

        // Calculate grid layout
        const cols = Math.min(this.featureMapCols, channels);
        const rows = Math.ceil(channels / cols);
        const mapSize = Math.min(
            this.featureMapSize,
            (width - 20) / cols - this.featureMapPadding,
            (height - 50) / rows - this.featureMapPadding
        );

        const startX = 10;
        const startY = 50;

        // Normalize data for visualization
        let minVal = Infinity, maxVal = -Infinity;
        for (let i = 0; i < data.length; i++) {
            minVal = Math.min(minVal, data[i]);
            maxVal = Math.max(maxVal, data[i]);
        }
        const range = maxVal - minVal || 1;

        // Draw each feature map
        for (let c = 0; c < channels; c++) {
            const col = c % cols;
            const row = Math.floor(c / cols);
            const x = startX + col * (mapSize + this.featureMapPadding);
            const y = startY + row * (mapSize + this.featureMapPadding);

            // Draw feature map
            this.drawFeatureMap(ctx, data, h, w, c, channels, x, y, mapSize, minVal, range);

            // Draw channel number
            ctx.fillStyle = '#64748b';
            ctx.font = '8px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(c.toString(), x + mapSize / 2, y + mapSize + 10);
        }
    }

    /**
     * Draw a single feature map
     */
    drawFeatureMap(ctx, data, h, w, channel, totalChannels, x, y, size, minVal, range) {
        const scaleX = size / w;
        const scaleY = size / h;

        // Draw border
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 1, y - 1, size + 2, size + 2);

        // Draw pixels
        for (let py = 0; py < h; py++) {
            for (let px = 0; px < w; px++) {
                const idx = (py * w + px) * totalChannels + channel;
                const val = (data[idx] - minVal) / range;

                // Use a blue-to-red colormap
                const color = this.valueToColor(val);
                ctx.fillStyle = color;
                ctx.fillRect(
                    x + px * scaleX,
                    y + py * scaleY,
                    Math.ceil(scaleX),
                    Math.ceil(scaleY)
                );
            }
        }
    }

    /**
     * Render dense layer activations as bar chart
     */
    async renderDenseActivations(ctx, activations, width, height) {
        const [batch, units] = activations.shape;
        const data = await activations.data();

        const startX = 20;
        const startY = 60;
        const barWidth = Math.min(20, (width - 40) / units - 2);
        const maxBarHeight = height - 100;

        // Find min/max for scaling
        let minVal = Infinity, maxVal = -Infinity;
        for (let i = 0; i < units; i++) {
            minVal = Math.min(minVal, data[i]);
            maxVal = Math.max(maxVal, data[i]);
        }
        const range = maxVal - minVal || 1;

        // Draw bars
        for (let i = 0; i < Math.min(units, 50); i++) {
            const val = (data[i] - minVal) / range;
            const barHeight = val * maxBarHeight;
            const x = startX + i * (barWidth + 2);
            const y = startY + maxBarHeight - barHeight;

            // Gradient fill
            const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
            gradient.addColorStop(0, this.valueToColor(val));
            gradient.addColorStop(1, this.darkenColor(this.valueToColor(val), 0.3));

            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, barWidth, barHeight);

            // Unit index
            if (units <= 20) {
                ctx.fillStyle = '#64748b';
                ctx.font = '8px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(i.toString(), x + barWidth / 2, startY + maxBarHeight + 12);
            }
        }

        // Indicate if truncated
        if (units > 50) {
            ctx.fillStyle = '#94a3b8';
            ctx.font = '10px system-ui, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(`Showing 50 of ${units} units`, startX, height - 10);
        }
    }

    /**
     * Convert value (0-1) to color
     */
    valueToColor(val) {
        // Blue (low) -> White (mid) -> Red (high)
        val = Math.max(0, Math.min(1, val));

        if (val < 0.5) {
            // Blue to white
            const t = val * 2;
            const r = Math.floor(59 + t * (255 - 59));
            const g = Math.floor(130 + t * (255 - 130));
            const b = 246;
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            // White to red
            const t = (val - 0.5) * 2;
            const r = 255;
            const g = Math.floor(255 - t * (255 - 68));
            const b = Math.floor(255 - t * (255 - 68));
            return `rgb(${r}, ${g}, ${b})`;
        }
    }

    /**
     * Render learned filters from a conv layer
     * @param {tf.Tensor[]} weights - Layer weights [kernel, bias]
     */
    async renderFilters(weights) {
        if (!this.featureCtx || !weights || weights.length === 0) return;

        const kernel = weights[0];  // Shape: [kernelH, kernelW, inputChannels, outputChannels]
        if (!kernel || kernel.shape.length !== 4) return;

        const canvas = this.featureCanvas;
        const ctx = this.featureCtx;
        const dpr = window.devicePixelRatio || 1;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const [kh, kw, inCh, outCh] = kernel.shape;
        const data = await kernel.data();

        // Calculate layout
        const filterSize = 30;
        const cols = Math.min(8, outCh);
        const rows = Math.ceil(outCh / cols);
        const startX = 10;
        const startY = 50;

        // Normalize
        let minVal = Infinity, maxVal = -Infinity;
        for (let i = 0; i < data.length; i++) {
            minVal = Math.min(minVal, data[i]);
            maxVal = Math.max(maxVal, data[i]);
        }
        const range = maxVal - minVal || 1;

        // Draw title
        ctx.fillStyle = '#f8fafc';
        ctx.font = '12px system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`Filters: ${kh}x${kw}x${inCh} → ${outCh} filters`, 10, 20);

        // Draw each filter (average across input channels)
        for (let f = 0; f < Math.min(outCh, 32); f++) {
            const col = f % cols;
            const row = Math.floor(f / cols);
            const x = startX + col * (filterSize + 6);
            const y = startY + row * (filterSize + 6);

            // Draw filter
            ctx.strokeStyle = '#334155';
            ctx.strokeRect(x - 1, y - 1, filterSize + 2, filterSize + 2);

            const scaleX = filterSize / kw;
            const scaleY = filterSize / kh;

            for (let ky = 0; ky < kh; ky++) {
                for (let kx = 0; kx < kw; kx++) {
                    // Average across input channels
                    let sum = 0;
                    for (let ic = 0; ic < inCh; ic++) {
                        const idx = ky * kw * inCh * outCh + kx * inCh * outCh + ic * outCh + f;
                        sum += data[idx];
                    }
                    const val = (sum / inCh - minVal) / range;

                    ctx.fillStyle = this.valueToColor(val);
                    ctx.fillRect(
                        x + kx * scaleX,
                        y + ky * scaleY,
                        Math.ceil(scaleX),
                        Math.ceil(scaleY)
                    );
                }
            }
        }
    }

    /**
     * Render input image
     * @param {Float32Array} imageData - Flat image data
     * @param {number} size - Image size (e.g., 28 for MNIST)
     * @param {number} channels - Number of channels
     */
    renderInputImage(ctx, imageData, size, channels, x, y, displaySize) {
        const scale = displaySize / size;

        // Draw border
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 2, y - 2, displaySize + 4, displaySize + 4);

        for (let py = 0; py < size; py++) {
            for (let px = 0; px < size; px++) {
                if (channels === 1) {
                    const val = imageData[py * size + px];
                    const gray = Math.floor(val * 255);
                    ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
                } else {
                    const idx = (py * size + px) * channels;
                    const r = Math.floor(imageData[idx] * 255);
                    const g = Math.floor(imageData[idx + 1] * 255);
                    const b = Math.floor(imageData[idx + 2] * 255);
                    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                }
                ctx.fillRect(x + px * scale, y + py * scale, Math.ceil(scale), Math.ceil(scale));
            }
        }
    }

    /**
     * Render prediction probabilities as bars
     * @param {CanvasRenderingContext2D} ctx
     * @param {Float32Array} probs - Prediction probabilities
     * @param {string[]} classNames - Class label names
     * @param {number} x - Start X
     * @param {number} y - Start Y
     * @param {number} width - Available width
     * @param {number} height - Available height
     */
    renderPredictionBars(ctx, probs, classNames, x, y, width, height) {
        const numClasses = probs.length;
        const barHeight = Math.min(20, (height - 20) / numClasses - 4);
        const maxBarWidth = width - 60;

        // Find predicted class
        let maxProb = 0, predictedClass = 0;
        for (let i = 0; i < numClasses; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                predictedClass = i;
            }
        }

        for (let i = 0; i < numClasses; i++) {
            const barY = y + i * (barHeight + 4);
            const prob = probs[i];
            const barWidth = prob * maxBarWidth;

            // Label
            ctx.fillStyle = i === predictedClass ? '#f8fafc' : '#94a3b8';
            ctx.font = i === predictedClass ? 'bold 11px system-ui, sans-serif' : '11px system-ui, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(classNames[i] || i.toString(), x + 35, barY + barHeight / 2 + 4);

            // Bar background
            ctx.fillStyle = '#1e293b';
            ctx.fillRect(x + 40, barY, maxBarWidth, barHeight);

            // Bar fill
            const color = i === predictedClass ? '#22c55e' : '#3b82f6';
            ctx.fillStyle = color;
            ctx.fillRect(x + 40, barY, barWidth, barHeight);

            // Probability text
            ctx.fillStyle = '#f8fafc';
            ctx.font = '10px monospace';
            ctx.textAlign = 'left';
            ctx.fillText(`${(prob * 100).toFixed(1)}%`, x + 45 + barWidth, barY + barHeight / 2 + 3);
        }
    }

    /**
     * Setup canvas for HiDPI displays
     * @param {HTMLCanvasElement} canvas
     */
    setupCanvas(canvas) {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        return ctx;
    }

    /**
     * Resize handler
     */
    resize() {
        if (this.archCanvas) {
            this.archCtx = this.setupCanvas(this.archCanvas);
        }
        if (this.featureCanvas) {
            this.featureCtx = this.setupCanvas(this.featureCanvas);
        }
        this.renderArchitecture();
    }

    /**
     * Clean up
     */
    dispose() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }
}
