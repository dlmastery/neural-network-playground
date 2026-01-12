/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Neural Network Visualization - TensorFlow Playground Style
 * Each neuron shows its decision boundary as a mini heatmap
 * ═══════════════════════════════════════════════════════════════════════════
 */

export class NetworkVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        this.config = {
            padding: 50,
            tileSize: 45,          // Size of each neuron tile
            tileSizeSmall: 32,     // Smaller tiles when many neurons
            maxNeuronsFullSize: 6,
            gridResolution: 12,    // Resolution of mini heatmaps
            colors: {
                classA: [245, 158, 11],    // Amber RGB
                classB: [34, 197, 94],     // Green RGB
                weight: {
                    positive: '#22c55e',
                    negative: '#f59e0b',
                    neutral: '#94a3b8'
                },
                border: {
                    input: '#22c55e',
                    hidden: '#f59e0b',
                    output: '#64748b'
                }
            },
            animation: {
                pulseSpeed: 0.015
            }
        };

        this.network = null;
        this.architecture = [];
        this.neuronPositions = [];
        this.neuronHeatmaps = [];  // Pre-computed heatmap images
        this.animationPhase = 0;
        this.hoveredNeuron = null;
        this.inputFeatures = null;  // Feature config from main app
        this.transformFeatures = null;  // Feature transform function

        this.init();
    }

    init() {
        this.resize();
        this.bindEvents();
        this.animate();
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        this.width = rect.width;
        this.height = rect.height;

        this.canvas.width = this.width * dpr;
        this.canvas.height = this.height * dpr;
        this.canvas.style.width = this.width + 'px';
        this.canvas.style.height = this.height + 'px';

        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.scale(dpr, dpr);

        if (this.architecture.length > 0) {
            this.calculatePositions();
        }
    }

    bindEvents() {
        window.addEventListener('resize', () => this.resize());

        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            this.handleMouseMove(x, y);
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.hoveredNeuron = null;
        });
    }

    handleMouseMove(x, y) {
        this.hoveredNeuron = null;
        for (let l = 0; l < this.neuronPositions.length; l++) {
            for (let n = 0; n < this.neuronPositions[l].length; n++) {
                const pos = this.neuronPositions[l][n];
                const size = this.getTileSize(l);
                if (x >= pos.x - size/2 && x <= pos.x + size/2 &&
                    y >= pos.y - size/2 && y <= pos.y + size/2) {
                    this.hoveredNeuron = { layer: l, neuron: n, pos };
                    return;
                }
            }
        }
    }

    getTileSize(layerIndex) {
        // Uniform tile size for all neurons
        return this.config.tileSize;
    }

    setArchitecture(architecture) {
        this.architecture = architecture;
        this.calculatePositions();
        this.neuronHeatmaps = architecture.map(size => new Array(size).fill(null));
    }

    setNetwork(network) {
        this.network = network;
        this.setArchitecture(network.architecture);
    }

    calculatePositions() {
        this.neuronPositions = [];
        const numLayers = this.architecture.length;
        if (numLayers === 0) return;

        const padding = this.config.padding;
        const usableWidth = this.width - padding * 2;
        const usableHeight = this.height - padding * 2;

        // Horizontal layout: layers spread across X axis
        const layerSpacing = usableWidth / Math.max(numLayers - 1, 1);

        for (let l = 0; l < numLayers; l++) {
            const layerPositions = [];
            const numNeurons = this.architecture[l];
            const tileSize = this.getTileSize(l);
            const totalHeight = numNeurons * tileSize + (numNeurons - 1) * 8;
            const startY = (this.height - totalHeight) / 2 + tileSize / 2;

            for (let n = 0; n < numNeurons; n++) {
                layerPositions.push({
                    x: padding + l * layerSpacing,
                    y: startY + n * (tileSize + 8)
                });
            }
            this.neuronPositions.push(layerPositions);
        }
    }

    /**
     * Update the mini heatmaps for all neurons
     * This is called periodically during training
     */
    updateNeuronHeatmaps() {
        if (!this.network) return;

        const resolution = this.config.gridResolution;
        const range = { min: -1.2, max: 1.2 };
        const step = (range.max - range.min) / resolution;

        // Generate input grid
        const inputs = [];
        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const x = range.min + j * step + step / 2;
                const y = range.max - i * step - step / 2;
                inputs.push([x, y]);
            }
        }

        // For each layer (except input), compute neuron outputs
        for (let l = 1; l < this.architecture.length; l++) {
            const numNeurons = this.architecture[l];

            for (let n = 0; n < numNeurons; n++) {
                // Create heatmap canvas for this neuron
                const heatmapCanvas = document.createElement('canvas');
                heatmapCanvas.width = resolution;
                heatmapCanvas.height = resolution;
                const heatmapCtx = heatmapCanvas.getContext('2d');
                const imageData = heatmapCtx.createImageData(resolution, resolution);

                // Get outputs for this specific neuron across all input points
                // Check if this is the output layer (uses sigmoid with 0-1 range)
                const isOutputLayer = l === this.architecture.length - 1;

                for (let i = 0; i < inputs.length; i++) {
                    const [x, y] = inputs[i];
                    const output = this.getNeuronOutput(l, n, x, y);

                    // Map output to color based on activation range
                    // Output layer (sigmoid): 0 to 1, Hidden layers (tanh): -1 to 1
                    let t;
                    if (isOutputLayer) {
                        t = output;  // Sigmoid already 0-1
                    } else {
                        t = (output + 1) / 2;  // Normalize tanh from -1,1 to 0,1
                    }
                    const clampedT = Math.max(0, Math.min(1, t));

                    const idx = i * 4;
                    const { classA, classB } = this.config.colors;
                    imageData.data[idx] = Math.round(classB[0] * (1 - clampedT) + classA[0] * clampedT);
                    imageData.data[idx + 1] = Math.round(classB[1] * (1 - clampedT) + classA[1] * clampedT);
                    imageData.data[idx + 2] = Math.round(classB[2] * (1 - clampedT) + classA[2] * clampedT);
                    imageData.data[idx + 3] = 255;
                }

                heatmapCtx.putImageData(imageData, 0, 0);
                this.neuronHeatmaps[l][n] = heatmapCanvas;
            }
        }
    }

    /**
     * Get the output of a specific neuron for given input
     * Uses transformed features if a transform function is provided
     *
     * layerIndex: 1 = first hidden layer, 2 = second hidden layer or output, etc.
     * network.layers[0] maps input -> first hidden layer
     * network.layers[n] maps layer n -> layer n+1
     */
    getNeuronOutput(layerIndex, neuronIndex, x1, x2) {
        if (!this.network) return 0;

        // Transform raw coordinates to selected features
        let input;
        if (this.transformFeatures) {
            input = this.transformFeatures(x1, x2);
        } else {
            input = [x1, x2];
        }

        // Forward pass through all layers BEFORE the target layer
        // layerIndex 1 needs layers[0], layerIndex 2 needs layers[0] then layers[1], etc.
        for (let l = 0; l < layerIndex - 1; l++) {
            const layer = this.network.layers[l];
            const output = [];

            for (let j = 0; j < layer.outputSize; j++) {
                let sum = layer.bias.data[j];
                for (let i = 0; i < layer.inputSize; i++) {
                    sum += input[i] * layer.weights.get(i, j);
                }
                output.push(this.applyActivation(sum, layer.activationName));
            }
            input = output;
        }

        // Now compute the specific neuron's output in the target layer
        const targetLayer = this.network.layers[layerIndex - 1];
        let sum = targetLayer.bias.data[neuronIndex];
        for (let i = 0; i < targetLayer.inputSize; i++) {
            sum += input[i] * targetLayer.weights.get(i, neuronIndex);
        }

        return this.applyActivation(sum, targetLayer.activationName);
    }

    applyActivation(x, activation) {
        switch (activation) {
            case 'relu': return Math.max(0, x);
            case 'sigmoid': return 1 / (1 + Math.exp(-x));
            case 'tanh': return Math.tanh(x);
            case 'leakyRelu': return x > 0 ? x : 0.01 * x;
            case 'swish': return x / (1 + Math.exp(-x));
            default: return x;
        }
    }

    updateActivations(activations) {
        // Update heatmaps every few frames for performance
        this.updateNeuronHeatmaps();
    }

    drawConnection(fromPos, toPos, weight, fromSize, toSize) {
        const normalizedWeight = Math.tanh(weight);
        const absWeight = Math.abs(normalizedWeight);

        let color;
        if (normalizedWeight > 0.01) {
            color = this.config.colors.weight.positive;
        } else if (normalizedWeight < -0.01) {
            color = this.config.colors.weight.negative;
        } else {
            color = this.config.colors.weight.neutral;
        }

        const lineWidth = 0.5 + absWeight * 2;
        const opacity = 0.1 + absWeight * 0.4;

        // Connect from right edge of source to left edge of target
        const startX = fromPos.x + fromSize / 2;
        const endX = toPos.x - toSize / 2;

        this.ctx.beginPath();
        this.ctx.moveTo(startX, fromPos.y);
        this.ctx.lineTo(endX, toPos.y);
        this.ctx.strokeStyle = color + Math.floor(opacity * 255).toString(16).padStart(2, '0');
        this.ctx.lineWidth = lineWidth;
        this.ctx.stroke();
    }

    drawNeuronTile(x, y, layerIndex, neuronIndex) {
        const isInput = layerIndex === 0;
        const isOutput = layerIndex === this.architecture.length - 1;
        const size = this.getTileSize(layerIndex);

        const isHovered = this.hoveredNeuron &&
            this.hoveredNeuron.layer === layerIndex &&
            this.hoveredNeuron.neuron === neuronIndex;

        const displaySize = isHovered ? size * 1.1 : size;

        // Draw heatmap or solid color for input layer
        if (isInput) {
            // Input neurons show gradient based on which input they represent
            const gradient = this.ctx.createLinearGradient(
                x - displaySize/2, y - displaySize/2,
                x + displaySize/2, y + displaySize/2
            );
            if (neuronIndex === 0) {
                // x1 - horizontal gradient
                gradient.addColorStop(0, '#22c55e');
                gradient.addColorStop(1, '#f59e0b');
            } else {
                // x2 - vertical gradient
                gradient.addColorStop(0, '#22c55e');
                gradient.addColorStop(1, '#f59e0b');
            }

            this.ctx.fillStyle = gradient;
            this.ctx.fillRect(x - displaySize/2, y - displaySize/2, displaySize, displaySize);
        } else {
            // Hidden/output neurons show their learned heatmap
            const heatmap = this.neuronHeatmaps[layerIndex]?.[neuronIndex];
            if (heatmap) {
                this.ctx.imageSmoothingEnabled = true;
                this.ctx.imageSmoothingQuality = 'high';
                this.ctx.drawImage(
                    heatmap,
                    x - displaySize/2,
                    y - displaySize/2,
                    displaySize,
                    displaySize
                );
            } else {
                // Fallback solid color
                this.ctx.fillStyle = '#e2e8f0';
                this.ctx.fillRect(x - displaySize/2, y - displaySize/2, displaySize, displaySize);
            }
        }

        // Draw border
        let borderColor;
        if (isInput) borderColor = this.config.colors.border.input;
        else if (isOutput) borderColor = this.config.colors.border.output;
        else borderColor = this.config.colors.border.hidden;

        this.ctx.strokeStyle = isHovered ? '#334155' : borderColor;
        this.ctx.lineWidth = isHovered ? 3 : 2;
        this.ctx.strokeRect(x - displaySize/2, y - displaySize/2, displaySize, displaySize);

        // Draw label below tile
        this.ctx.fillStyle = '#64748b';
        this.ctx.font = '10px Inter, sans-serif';
        this.ctx.textAlign = 'center';

        if (isInput) {
            const label = this.getInputFeatureLabel(neuronIndex);
            this.ctx.fillText(label, x, y + displaySize/2 + 14);
        }
    }

    /**
     * Get the label for an input neuron based on active features
     */
    getInputFeatureLabel(neuronIndex) {
        if (!this.inputFeatures) {
            return `x${neuronIndex + 1}`;
        }

        const featureLabels = {
            x1: 'x₁',
            x2: 'x₂',
            x1sq: 'x₁²',
            x2sq: 'x₂²',
            x1x2: 'x₁x₂',
            sinx1: 'sin',
            sinx2: 'cos'
        };

        const activeFeatures = Object.entries(this.inputFeatures)
            .filter(([_, active]) => active)
            .map(([key, _]) => featureLabels[key] || key);

        return activeFeatures[neuronIndex] || `f${neuronIndex + 1}`;
    }

    drawLayerLabel(layerIndex) {
        const x = this.neuronPositions[layerIndex][0].x;
        const y = 25;

        let label;
        if (layerIndex === 0) label = 'Input';
        else if (layerIndex === this.architecture.length - 1) label = 'Output';
        else label = `Hidden ${layerIndex}`;

        this.ctx.fillStyle = '#475569';
        this.ctx.font = '12px Inter, sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(label, x, y);
    }

    render() {
        this.ctx.clearRect(0, 0, this.width, this.height);

        if (this.architecture.length === 0) {
            this.drawEmptyState();
            return;
        }

        // Draw connections first (behind neurons)
        if (this.network) {
            const weights = this.network.getAllWeights();
            for (const w of weights) {
                const fromPos = this.neuronPositions[w.layerIndex][w.fromNeuron];
                const toPos = this.neuronPositions[w.layerIndex + 1][w.toNeuron];
                if (fromPos && toPos) {
                    const fromSize = this.getTileSize(w.layerIndex);
                    const toSize = this.getTileSize(w.layerIndex + 1);
                    this.drawConnection(fromPos, toPos, w.weight, fromSize, toSize);
                }
            }
        }

        // Draw neuron tiles
        for (let l = 0; l < this.architecture.length; l++) {
            this.drawLayerLabel(l);
            for (let n = 0; n < this.architecture[l]; n++) {
                const pos = this.neuronPositions[l][n];
                this.drawNeuronTile(pos.x, pos.y, l, n);
            }
        }
    }

    drawEmptyState() {
        this.ctx.fillStyle = '#94a3b8';
        this.ctx.font = '13px Inter, sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText('Add layers to build your network', this.width / 2, this.height / 2);
    }

    animate() {
        this.animationPhase = (this.animationPhase + this.config.animation.pulseSpeed) % 1;
        this.render();
        requestAnimationFrame(() => this.animate());
    }
}
