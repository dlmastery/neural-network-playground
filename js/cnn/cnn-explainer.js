/**
 * CNN Explainer
 * Interactive visualization to help users understand how CNNs work
 * Includes: filter visualization, activation maps, saliency, convolution animation
 */

export class CNNExplainer {
    constructor(modelBuilder, datasetLoader) {
        this.modelBuilder = modelBuilder;
        this.datasetLoader = datasetLoader;

        // UI elements
        this.modal = null;
        this.currentSample = null;
        this.currentLayer = 0;

        // Animation state
        this.convAnimationFrame = null;
        this.isAnimating = false;

        // Canvas contexts
        this.contexts = {};

        // Computed data
        this.activations = [];
        this.saliencyMap = null;

        // Drawing state
        this.isDrawing = false;
        this.lastDrawPos = { x: 0, y: 0 };

        // Softmax data
        this.logits = null;
    }

    /**
     * Initialize the explainer modal
     */
    init() {
        this.modal = document.getElementById('cnn-explainer-modal');
        if (!this.modal) return;

        // Cache canvas contexts
        this.contexts = {
            input: document.getElementById('explainer-input-canvas')?.getContext('2d'),
            filters: document.getElementById('explainer-filters-canvas')?.getContext('2d'),
            activations: document.getElementById('explainer-activations-canvas')?.getContext('2d'),
            saliency: document.getElementById('explainer-saliency-canvas')?.getContext('2d'),
            convAnim: document.getElementById('explainer-conv-anim-canvas')?.getContext('2d'),
            // New educational canvases
            drawCanvas: document.getElementById('explainer-draw-canvas')?.getContext('2d'),
            drawProcessed: document.getElementById('explainer-draw-processed')?.getContext('2d'),
            flattenBefore: document.getElementById('explainer-flatten-before')?.getContext('2d'),
            flattenAfter: document.getElementById('explainer-flatten-after')?.getContext('2d'),
            softmaxLogits: document.getElementById('explainer-softmax-logits')?.getContext('2d'),
            softmaxProbs: document.getElementById('explainer-softmax-probs')?.getContext('2d')
        };

        this.setupEventListeners();
        this.setupDrawingCanvas();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Close button
        document.getElementById('explainer-close-btn')?.addEventListener('click', () => {
            this.close();
        });

        // All layer selectors (sync them)
        const layerSelectors = [
            document.getElementById('explainer-layer-select'),
            document.getElementById('explainer-filters-layer-select'),
            document.getElementById('explainer-activations-layer-select')
        ];

        layerSelectors.forEach(select => {
            select?.addEventListener('change', (e) => {
                this.currentLayer = parseInt(e.target.value);
                // Sync all selectors
                layerSelectors.forEach(s => {
                    if (s && s !== e.target) s.value = this.currentLayer;
                });
                this.updateLayerVisualization();
            });
        });

        // New sample button
        document.getElementById('explainer-new-sample-btn')?.addEventListener('click', () => {
            this.loadNewSample();
        });

        // Play convolution animation
        document.getElementById('explainer-play-conv-btn')?.addEventListener('click', () => {
            this.toggleConvAnimation();
        });

        // Drawing canvas buttons
        document.getElementById('explainer-clear-draw-btn')?.addEventListener('click', () => {
            this.clearDrawingCanvas();
        });

        document.getElementById('explainer-predict-draw-btn')?.addEventListener('click', () => {
            this.predictDrawing();
        });

        // Tab switching
        document.querySelectorAll('.explainer-tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Close on backdrop click
        this.modal?.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.close();
            }
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.modal?.classList.contains('active')) {
                this.close();
            }
        });
    }

    /**
     * Open the explainer modal
     */
    async open() {
        if (!this.modal) return;
        if (!this.modelBuilder.model) {
            alert('Please train the model first!');
            return;
        }

        this.modal.classList.add('active');
        document.body.style.overflow = 'hidden';

        // Populate layer selector
        this.populateLayerSelector();

        // Load initial sample
        await this.loadNewSample();
    }

    /**
     * Close the explainer modal
     */
    close() {
        this.modal?.classList.remove('active');
        document.body.style.overflow = '';
        this.stopConvAnimation();

        // Dispose tensors
        this.disposeActivations();
        if (this.saliencyMap) {
            this.saliencyMap.dispose();
            this.saliencyMap = null;
        }
    }

    /**
     * Populate all layer selector dropdowns
     */
    populateLayerSelector() {
        const selectors = [
            document.getElementById('explainer-layer-select'),
            document.getElementById('explainer-filters-layer-select'),
            document.getElementById('explainer-activations-layer-select')
        ];

        if (!this.modelBuilder.model) return;

        selectors.forEach(select => {
            if (!select) return;

            // Clear existing options using safe method
            select.textContent = '';

            this.modelBuilder.layers.forEach((layer, i) => {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `${i + 1}. ${this.getLayerDisplayName(layer)}`;
                select.appendChild(option);
            });
        });

        this.currentLayer = 0;
    }

    /**
     * Get display name for layer
     */
    getLayerDisplayName(layer) {
        const names = {
            conv2d: 'Conv2D',
            maxPooling2d: 'MaxPool2D',
            avgPooling2d: 'AvgPool2D',
            batchNormalization: 'BatchNorm',
            dropout: 'Dropout',
            flatten: 'Flatten',
            dense: 'Dense'
        };

        let name = names[layer.type] || layer.type;

        if (layer.type === 'conv2d') {
            name += ` (${layer.config.filters} filters, ${layer.config.kernelSize}x${layer.config.kernelSize})`;
        } else if (layer.type === 'dense') {
            name += ` (${layer.config.units} units)`;
        }

        return name;
    }

    /**
     * Load a new random sample
     */
    async loadNewSample() {
        if (!this.datasetLoader.isLoaded) return;

        // Get random sample
        const sample = this.datasetLoader.getRandomSample(false);
        if (!sample) return;

        // Store current sample
        if (this.currentSample) {
            this.currentSample.image.dispose();
        }
        this.currentSample = sample;

        // Compute all activations
        await this.computeActivations();

        // Compute saliency map
        await this.computeSaliency();

        // Update all visualizations
        this.renderInputImage();
        this.updateLayerVisualization();
        this.renderSaliencyMap();
        this.renderPrediction();
    }

    /**
     * Compute activations at all layers
     */
    async computeActivations() {
        this.disposeActivations();

        if (!this.currentSample || !this.modelBuilder.model) return;

        this.activations = this.modelBuilder.getAllActivations(this.currentSample.image);
    }

    /**
     * Dispose all activation tensors
     */
    disposeActivations() {
        this.activations.forEach(act => {
            if (act) act.dispose();
        });
        this.activations = [];
    }

    /**
     * Compute saliency map using gradient-based method
     */
    async computeSaliency() {
        if (!this.currentSample || !this.modelBuilder.model) return;

        if (this.saliencyMap) {
            this.saliencyMap.dispose();
            this.saliencyMap = null;
        }

        try {
            // Use gradient-based saliency
            const saliency = tf.tidy(() => {
                const input = this.currentSample.image;

                // Compute gradients of max output with respect to input
                const gradFunc = (x) => {
                    const output = this.modelBuilder.model.apply(x, { training: false });
                    return output.max(1);  // Max class score
                };

                const grad = tf.grad(gradFunc);
                const gradients = grad(input);

                // Take absolute value and max across channels
                const absGrad = gradients.abs();
                const saliencyMap = absGrad.max(-1);  // Max across channels

                // Normalize to 0-1
                const minVal = saliencyMap.min();
                const maxVal = saliencyMap.max();
                const range = maxVal.sub(minVal).add(1e-8);
                return saliencyMap.sub(minVal).div(range);
            });

            this.saliencyMap = saliency;
        } catch (e) {
            console.warn('Could not compute saliency:', e.message);
        }
    }

    /**
     * Render the input image
     */
    renderInputImage() {
        const ctx = this.contexts.input;
        if (!ctx || !this.currentSample) return;

        const canvas = ctx.canvas;
        const config = this.datasetLoader.getConfig();

        // Clear canvas
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw input image centered
        const imgSize = Math.min(canvas.width, canvas.height) - 40;
        const x = (canvas.width - imgSize) / 2;
        const y = (canvas.height - imgSize) / 2;

        this.drawImageData(
            ctx,
            this.currentSample.imageData,
            config.imageSize,
            config.channels,
            x, y, imgSize
        );

        // Draw label
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 14px Inter, sans-serif';
        ctx.textAlign = 'center';
        const label = config.classNames[this.currentSample.label] || `Class ${this.currentSample.label}`;
        ctx.fillText(`True: ${label}`, canvas.width / 2, canvas.height - 10);
    }

    /**
     * Draw image data to canvas
     */
    drawImageData(ctx, data, size, channels, x, y, displaySize) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = size;
        tempCanvas.height = size;
        const tempCtx = tempCanvas.getContext('2d');
        const imageData = tempCtx.createImageData(size, size);

        for (let i = 0; i < size * size; i++) {
            const pixelIndex = i * 4;

            if (channels === 1) {
                const val = Math.floor(data[i] * 255);
                imageData.data[pixelIndex] = val;
                imageData.data[pixelIndex + 1] = val;
                imageData.data[pixelIndex + 2] = val;
            } else {
                imageData.data[pixelIndex] = Math.floor(data[i * 3] * 255);
                imageData.data[pixelIndex + 1] = Math.floor(data[i * 3 + 1] * 255);
                imageData.data[pixelIndex + 2] = Math.floor(data[i * 3 + 2] * 255);
            }
            imageData.data[pixelIndex + 3] = 255;
        }

        tempCtx.putImageData(imageData, 0, 0);

        // Draw scaled with pixelated rendering
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, x, y, displaySize, displaySize);

        // Border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, displaySize, displaySize);
    }

    /**
     * Update visualization for selected layer
     */
    async updateLayerVisualization() {
        this.renderFilters();
        this.renderActivations();
        this.updateLayerInfo();
    }

    /**
     * Render conv filters for current layer
     */
    async renderFilters() {
        const ctx = this.contexts.filters;
        if (!ctx) return;

        const canvas = ctx.canvas;
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Get layer weights
        const layer = this.modelBuilder.layers[this.currentLayer];
        if (!layer || layer.type !== 'conv2d') {
            ctx.fillStyle = '#64748b';
            ctx.font = '14px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Filters only available for Conv2D layers', canvas.width / 2, canvas.height / 2);
            return;
        }

        const weights = this.modelBuilder.getLayerWeights(this.currentLayer);
        if (!weights || weights.length === 0) return;

        // Weights shape: [kernelH, kernelW, inputChannels, outputFilters]
        const kernelTensor = weights[0];
        const kernelData = await kernelTensor.data();
        const [kH, kW, inputChannels, numFilters] = kernelTensor.shape;

        // Calculate grid layout
        const cols = Math.ceil(Math.sqrt(numFilters));
        const rows = Math.ceil(numFilters / cols);
        const padding = 4;
        const filterSize = Math.min(
            (canvas.width - padding * (cols + 1)) / cols,
            (canvas.height - 40 - padding * (rows + 1)) / rows
        );

        // Title
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${numFilters} Filters (${kH}x${kW})`, canvas.width / 2, 20);

        const startY = 35;

        // Draw each filter
        for (let f = 0; f < numFilters; f++) {
            const col = f % cols;
            const row = Math.floor(f / cols);
            const x = padding + col * (filterSize + padding);
            const y = startY + padding + row * (filterSize + padding);

            // Extract filter weights (average across input channels for visualization)
            const filterData = new Float32Array(kH * kW);
            for (let i = 0; i < kH; i++) {
                for (let j = 0; j < kW; j++) {
                    let sum = 0;
                    for (let c = 0; c < inputChannels; c++) {
                        const idx = i * kW * inputChannels * numFilters +
                                    j * inputChannels * numFilters +
                                    c * numFilters + f;
                        sum += kernelData[idx];
                    }
                    filterData[i * kW + j] = sum / inputChannels;
                }
            }

            // Normalize filter data
            let minVal = Infinity, maxVal = -Infinity;
            for (let i = 0; i < filterData.length; i++) {
                minVal = Math.min(minVal, filterData[i]);
                maxVal = Math.max(maxVal, filterData[i]);
            }
            const range = maxVal - minVal || 1;

            // Draw filter as heatmap
            this.drawHeatmap(ctx, filterData, kW, kH, x, y, filterSize, minVal, range);

            // Filter index label
            ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.font = '9px JetBrains Mono, monospace';
            ctx.textAlign = 'left';
            ctx.fillText(`${f}`, x + 2, y + 10);
        }
    }

    /**
     * Render activations (feature maps) for current layer
     */
    async renderActivations() {
        const ctx = this.contexts.activations;
        if (!ctx) return;

        const canvas = ctx.canvas;
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const activation = this.activations[this.currentLayer];
        if (!activation) {
            ctx.fillStyle = '#64748b';
            ctx.font = '14px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No activation data available', canvas.width / 2, canvas.height / 2);
            return;
        }

        const shape = activation.shape;
        const data = await activation.data();

        // Different rendering based on tensor shape
        if (shape.length === 4) {
            // Conv/Pool layer: [batch, height, width, channels]
            const [, h, w, channels] = shape;
            this.renderFeatureMapsGrid(ctx, data, w, h, channels, canvas);
        } else if (shape.length === 2) {
            // Dense layer: [batch, units]
            const units = shape[1];
            this.renderDenseActivations(ctx, data, units, canvas);
        }
    }

    /**
     * Render feature maps as grid
     */
    renderFeatureMapsGrid(ctx, data, w, h, channels, canvas) {
        // Title
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${channels} Feature Maps (${w}x${h})`, canvas.width / 2, 20);

        const cols = Math.ceil(Math.sqrt(channels));
        const rows = Math.ceil(channels / cols);
        const padding = 3;
        const mapSize = Math.min(
            (canvas.width - padding * (cols + 1)) / cols,
            (canvas.height - 40 - padding * (rows + 1)) / rows
        );

        const startY = 35;

        for (let c = 0; c < channels; c++) {
            const col = c % cols;
            const row = Math.floor(c / cols);
            const x = padding + col * (mapSize + padding);
            const y = startY + padding + row * (mapSize + padding);

            // Extract channel data
            const channelData = new Float32Array(h * w);
            for (let i = 0; i < h; i++) {
                for (let j = 0; j < w; j++) {
                    const idx = i * w * channels + j * channels + c;
                    channelData[i * w + j] = data[idx];
                }
            }

            // Find min/max for normalization
            let minVal = Infinity, maxVal = -Infinity;
            for (let i = 0; i < channelData.length; i++) {
                minVal = Math.min(minVal, channelData[i]);
                maxVal = Math.max(maxVal, channelData[i]);
            }
            const range = maxVal - minVal || 1;

            // Draw as heatmap
            this.drawHeatmap(ctx, channelData, w, h, x, y, mapSize, minVal, range, true);
        }
    }

    /**
     * Render dense layer activations as bar chart
     */
    renderDenseActivations(ctx, data, units, canvas) {
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${units} Neurons`, canvas.width / 2, 20);

        const padding = 20;
        const barWidth = Math.max(2, (canvas.width - padding * 2) / units);
        const maxHeight = canvas.height - 60;

        // Find max for normalization
        let maxVal = 0;
        for (let i = 0; i < units; i++) {
            maxVal = Math.max(maxVal, Math.abs(data[i]));
        }
        maxVal = maxVal || 1;

        for (let i = 0; i < units; i++) {
            const val = data[i];
            const normalizedVal = val / maxVal;
            const barHeight = Math.abs(normalizedVal) * maxHeight;

            const x = padding + i * barWidth;
            const y = val >= 0 ? canvas.height - 30 - barHeight : canvas.height - 30;

            // Color based on value
            const hue = val >= 0 ? 120 : 0;  // Green for positive, red for negative
            const saturation = Math.min(100, Math.abs(normalizedVal) * 100);
            ctx.fillStyle = `hsl(${hue}, ${saturation}%, 50%)`;

            ctx.fillRect(x, y, Math.max(1, barWidth - 1), barHeight);
        }

        // Zero line
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.beginPath();
        ctx.moveTo(padding, canvas.height - 30);
        ctx.lineTo(canvas.width - padding, canvas.height - 30);
        ctx.stroke();
    }

    /**
     * Draw a heatmap from 2D data
     */
    drawHeatmap(ctx, data, w, h, x, y, displaySize, minVal, range, useViridis = false) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = w;
        tempCanvas.height = h;
        const tempCtx = tempCanvas.getContext('2d');
        const imageData = tempCtx.createImageData(w, h);

        for (let i = 0; i < h; i++) {
            for (let j = 0; j < w; j++) {
                const idx = (i * w + j);
                const pixelIdx = idx * 4;
                const val = (data[idx] - minVal) / range;

                let r, g, b;
                if (useViridis) {
                    // Viridis-like colormap
                    [r, g, b] = this.viridisColor(val);
                } else {
                    // Blue-white-red diverging colormap
                    [r, g, b] = this.divergingColor(val);
                }

                imageData.data[pixelIdx] = r;
                imageData.data[pixelIdx + 1] = g;
                imageData.data[pixelIdx + 2] = b;
                imageData.data[pixelIdx + 3] = 255;
            }
        }

        tempCtx.putImageData(imageData, 0, 0);

        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, x, y, displaySize, displaySize);
    }

    /**
     * Viridis-like colormap
     */
    viridisColor(t) {
        // Simplified viridis approximation
        const r = Math.floor(68 + t * (253 - 68));
        const g = Math.floor(1 + t * 150 + (1 - t) * 84);
        const b = Math.floor(84 + t * (231 - 84) * (1 - t * 0.5));
        return [r, g, b];
    }

    /**
     * Blue-white-red diverging colormap
     */
    divergingColor(t) {
        if (t < 0.5) {
            // Blue to white
            const factor = t * 2;
            return [
                Math.floor(59 + factor * 196),
                Math.floor(130 + factor * 125),
                Math.floor(246 + factor * 9)
            ];
        } else {
            // White to red
            const factor = (t - 0.5) * 2;
            return [
                Math.floor(255 - factor * 16),
                Math.floor(255 - factor * 172),
                Math.floor(255 - factor * 187)
            ];
        }
    }

    /**
     * Render saliency map
     */
    async renderSaliencyMap() {
        const ctx = this.contexts.saliency;
        if (!ctx || !this.saliencyMap || !this.currentSample) return;

        const canvas = ctx.canvas;
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const config = this.datasetLoader.getConfig();
        const imgSize = Math.min(canvas.width, canvas.height) - 40;
        const x = (canvas.width - imgSize) / 2;
        const y = 20;

        // Get saliency data
        const saliencyData = await this.saliencyMap.data();
        const [, h, w] = this.saliencyMap.shape;

        // Draw input image first (as background)
        ctx.globalAlpha = 0.4;
        this.drawImageData(ctx, this.currentSample.imageData, config.imageSize, config.channels, x, y, imgSize);
        ctx.globalAlpha = 1.0;

        // Overlay saliency heatmap
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = w;
        tempCanvas.height = h;
        const tempCtx = tempCanvas.getContext('2d');
        const imageData = tempCtx.createImageData(w, h);

        for (let i = 0; i < h; i++) {
            for (let j = 0; j < w; j++) {
                const idx = i * w + j;
                const pixelIdx = idx * 4;
                const val = saliencyData[idx];

                // Hot colormap
                const r = Math.min(255, Math.floor(val * 255 * 2));
                const g = Math.min(255, Math.floor(Math.max(0, val - 0.5) * 255 * 2));
                const b = 0;

                imageData.data[pixelIdx] = r;
                imageData.data[pixelIdx + 1] = g;
                imageData.data[pixelIdx + 2] = b;
                imageData.data[pixelIdx + 3] = Math.floor(val * 200);  // Alpha based on saliency
            }
        }

        tempCtx.putImageData(imageData, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, x, y, imgSize, imgSize);

        // Title
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Saliency Map (Important Regions)', canvas.width / 2, canvas.height - 8);
    }

    /**
     * Render prediction bars using safe DOM methods
     */
    async renderPrediction() {
        const container = document.getElementById('explainer-prediction');
        if (!container || !this.currentSample || !this.modelBuilder.model) return;

        // Get prediction
        const pred = this.modelBuilder.predict(this.currentSample.image);
        const probs = await pred.data();
        pred.dispose();

        const config = this.datasetLoader.getConfig();

        // Find top prediction
        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) {
            if (probs[i] > probs[maxIdx]) maxIdx = i;
        }

        // Clear container using safe method
        container.textContent = '';

        // Sort by probability
        const sorted = Array.from(probs)
            .map((p, i) => ({ prob: p, idx: i }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 5);  // Top 5

        sorted.forEach(({ prob, idx }) => {
            const isCorrect = idx === this.currentSample.label;
            const isTop = idx === maxIdx;

            const bar = document.createElement('div');
            bar.className = 'explainer-pred-bar';

            const label = document.createElement('span');
            label.className = 'explainer-pred-bar__label';
            label.textContent = config.classNames[idx] || `Class ${idx}`;

            const track = document.createElement('div');
            track.className = 'explainer-pred-bar__track';

            const fill = document.createElement('div');
            fill.className = 'explainer-pred-bar__fill';
            if (isCorrect) fill.classList.add('explainer-pred-bar__fill--correct');
            if (isTop && !isCorrect) fill.classList.add('explainer-pred-bar__fill--top');
            fill.style.width = `${prob * 100}%`;

            const value = document.createElement('span');
            value.className = 'explainer-pred-bar__value';
            value.textContent = `${(prob * 100).toFixed(1)}%`;

            track.appendChild(fill);
            bar.appendChild(label);
            bar.appendChild(track);
            bar.appendChild(value);
            container.appendChild(bar);
        });

        // Result summary
        const summary = document.createElement('div');
        summary.className = 'explainer-pred-summary';
        const isCorrect = maxIdx === this.currentSample.label;
        summary.classList.add(isCorrect ? 'explainer-pred-summary--correct' : 'explainer-pred-summary--wrong');
        summary.textContent = isCorrect ? 'Correct Prediction!' : `Wrong! (Predicted: ${config.classNames[maxIdx]})`;
        container.appendChild(summary);
    }

    /**
     * Update layer info panel
     */
    updateLayerInfo() {
        const infoEl = document.getElementById('explainer-layer-info');
        if (!infoEl) return;

        const layer = this.modelBuilder.layers[this.currentLayer];
        const tfLayer = this.modelBuilder.model?.layers[this.currentLayer];

        if (!layer || !tfLayer) {
            infoEl.textContent = 'No layer info available';
            return;
        }

        const outputShape = tfLayer.outputShape;
        const weights = tfLayer.getWeights();
        let params = 0;
        weights.forEach(w => params += w.size);

        let info = `Type: ${this.getLayerDisplayName(layer)}\n`;
        info += `Output Shape: [${outputShape.join(', ')}]\n`;
        info += `Parameters: ${params.toLocaleString()}`;

        if (layer.type === 'conv2d') {
            info += `\nActivation: ${layer.config.activation}`;
            info += `\nPadding: ${layer.config.padding}`;
        } else if (layer.type === 'dropout') {
            info += `\nRate: ${(layer.config.rate * 100).toFixed(0)}%`;
        }

        infoEl.textContent = info;
    }

    /**
     * Switch between explainer tabs
     */
    switchTab(tabId) {
        // Update tab buttons
        document.querySelectorAll('.explainer-tab-btn').forEach(btn => {
            btn.classList.toggle('explainer-tab-btn--active', btn.dataset.tab === tabId);
        });

        // Update tab panels
        document.querySelectorAll('.explainer-tab-panel').forEach(panel => {
            panel.classList.toggle('explainer-tab-panel--active', panel.id === `explainer-${tabId}-panel`);
        });

        // Trigger specific actions for tabs
        if (tabId === 'conv-anim') {
            this.setupConvAnimation();
        } else if (tabId === 'draw') {
            this.initDrawingCanvas();
        } else if (tabId === 'flatten') {
            this.renderFlattenVisualization();
        } else if (tabId === 'softmax') {
            this.renderSoftmaxVisualization();
        }
    }

    /**
     * Setup convolution animation
     */
    setupConvAnimation() {
        const ctx = this.contexts.convAnim;
        if (!ctx || !this.currentSample) return;

        // Find first conv layer
        let convLayerIdx = -1;
        for (let i = 0; i < this.modelBuilder.layers.length; i++) {
            if (this.modelBuilder.layers[i].type === 'conv2d') {
                convLayerIdx = i;
                break;
            }
        }

        if (convLayerIdx < 0) {
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            ctx.fillStyle = '#64748b';
            ctx.font = '14px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No Conv2D layer in model', ctx.canvas.width / 2, ctx.canvas.height / 2);
            return;
        }

        this.convAnimLayerIdx = convLayerIdx;
        this.convAnimPos = { row: 0, col: 0 };
        this.renderConvAnimationFrame();
    }

    /**
     * Toggle convolution animation
     */
    toggleConvAnimation() {
        if (this.isAnimating) {
            this.stopConvAnimation();
        } else {
            this.startConvAnimation();
        }
    }

    /**
     * Start convolution animation
     */
    startConvAnimation() {
        this.isAnimating = true;
        const btn = document.getElementById('explainer-play-conv-btn');
        if (btn) btn.textContent = 'Pause';
        this.animateConvolution();
    }

    /**
     * Stop convolution animation
     */
    stopConvAnimation() {
        this.isAnimating = false;
        const btn = document.getElementById('explainer-play-conv-btn');
        if (btn) btn.textContent = 'Play';
        if (this.convAnimationFrame) {
            clearTimeout(this.convAnimationFrame);
            this.convAnimationFrame = null;
        }
    }

    /**
     * Animate convolution operation
     */
    animateConvolution() {
        if (!this.isAnimating) return;

        this.renderConvAnimationFrame();

        // Move position
        const config = this.datasetLoader.getConfig();
        const layer = this.modelBuilder.layers[this.convAnimLayerIdx];
        const kernelSize = layer?.config.kernelSize || 3;
        const stride = layer?.config.strides || 1;
        const inputSize = config.imageSize;
        const outputSize = inputSize - kernelSize + 1;  // Valid padding approximation

        this.convAnimPos.col += stride;
        if (this.convAnimPos.col >= outputSize) {
            this.convAnimPos.col = 0;
            this.convAnimPos.row += stride;
            if (this.convAnimPos.row >= outputSize) {
                this.convAnimPos.row = 0;
            }
        }

        this.convAnimationFrame = setTimeout(() => {
            requestAnimationFrame(() => this.animateConvolution());
        }, 100);  // 10fps for smooth visualization
    }

    /**
     * Render single frame of convolution animation
     */
    async renderConvAnimationFrame() {
        const ctx = this.contexts.convAnim;
        if (!ctx || !this.currentSample) return;

        const canvas = ctx.canvas;
        const config = this.datasetLoader.getConfig();
        const layer = this.modelBuilder.layers[this.convAnimLayerIdx];
        if (!layer) return;

        const kernelSize = layer.config.kernelSize || 3;

        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Layout
        const inputSize = 150;
        const filterSize = 80;
        const outputSize = 100;
        const spacing = 40;

        const inputX = 30;
        const filterX = inputX + inputSize + spacing;
        const outputX = filterX + filterSize + spacing;
        const centerY = canvas.height / 2;

        // Draw input image
        ctx.fillStyle = '#94a3b8';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Input', inputX + inputSize / 2, centerY - inputSize / 2 - 10);

        this.drawImageData(
            ctx, this.currentSample.imageData,
            config.imageSize, config.channels,
            inputX, centerY - inputSize / 2, inputSize
        );

        // Draw filter kernel position
        const scale = inputSize / config.imageSize;
        const kx = inputX + this.convAnimPos.col * scale;
        const ky = centerY - inputSize / 2 + this.convAnimPos.row * scale;
        const kSize = kernelSize * scale;

        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 2;
        ctx.strokeRect(kx, ky, kSize, kSize);

        // Draw filter
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(`Filter (${kernelSize}x${kernelSize})`, filterX + filterSize / 2, centerY - filterSize / 2 - 10);

        const weights = this.modelBuilder.getLayerWeights(this.convAnimLayerIdx);
        if (weights && weights.length > 0) {
            const kernelData = await weights[0].data();
            const [kH, kW, inputChannels, numFilters] = weights[0].shape;

            // Draw first filter
            const filterData = new Float32Array(kH * kW);
            for (let i = 0; i < kH; i++) {
                for (let j = 0; j < kW; j++) {
                    let sum = 0;
                    for (let c = 0; c < inputChannels; c++) {
                        const idx = i * kW * inputChannels * numFilters + j * inputChannels * numFilters + c * numFilters;
                        sum += kernelData[idx];
                    }
                    filterData[i * kW + j] = sum / inputChannels;
                }
            }

            let minVal = Infinity, maxVal = -Infinity;
            for (let i = 0; i < filterData.length; i++) {
                minVal = Math.min(minVal, filterData[i]);
                maxVal = Math.max(maxVal, filterData[i]);
            }

            this.drawHeatmap(ctx, filterData, kW, kH, filterX, centerY - filterSize / 2, filterSize, minVal, maxVal - minVal);
        }

        // Draw output feature map
        ctx.fillStyle = '#94a3b8';
        ctx.fillText('Output', outputX + outputSize / 2, centerY - outputSize / 2 - 10);

        const activation = this.activations[this.convAnimLayerIdx];
        if (activation && activation.shape.length === 4) {
            const actData = await activation.data();
            const [, h, w, c] = activation.shape;

            // Extract first channel
            const channelData = new Float32Array(h * w);
            for (let i = 0; i < h; i++) {
                for (let j = 0; j < w; j++) {
                    channelData[i * w + j] = actData[i * w * c + j * c];
                }
            }

            let minVal = Infinity, maxVal = -Infinity;
            for (let i = 0; i < channelData.length; i++) {
                minVal = Math.min(minVal, channelData[i]);
                maxVal = Math.max(maxVal, channelData[i]);
            }

            this.drawHeatmap(ctx, channelData, w, h, outputX, centerY - outputSize / 2, outputSize, minVal, maxVal - minVal, true);

            // Highlight current output position
            const outScale = outputSize / w;
            const outX = outputX + this.convAnimPos.col * outScale;
            const outY = centerY - outputSize / 2 + this.convAnimPos.row * outScale;
            ctx.strokeStyle = '#22c55e';
            ctx.lineWidth = 2;
            ctx.strokeRect(outX, outY, outScale, outScale);
        }

        // Draw arrows
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 2;

        // Arrow from input to filter
        this.drawArrow(ctx, inputX + inputSize + 10, centerY, filterX - 10, centerY);

        // Arrow from filter to output
        this.drawArrow(ctx, filterX + filterSize + 10, centerY, outputX - 10, centerY);

        // Convolution symbol
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 20px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('*', filterX - spacing / 2, centerY + 6);
        ctx.fillText('=', outputX - spacing / 2, centerY + 6);
    }

    /**
     * Draw arrow
     */
    drawArrow(ctx, fromX, fromY, toX, toY) {
        const headLen = 8;
        const angle = Math.atan2(toY - fromY, toX - fromX);

        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(toX - headLen * Math.cos(angle - Math.PI / 6), toY - headLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(toX - headLen * Math.cos(angle + Math.PI / 6), toY - headLen * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INTERACTIVE DRAWING CANVAS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Setup drawing canvas event listeners
     */
    setupDrawingCanvas() {
        const canvas = document.getElementById('explainer-draw-canvas');
        if (!canvas) return;

        // Mouse events
        canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        canvas.addEventListener('mousemove', (e) => this.draw(e));
        canvas.addEventListener('mouseup', () => this.stopDrawing());
        canvas.addEventListener('mouseout', () => this.stopDrawing());

        // Touch events
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startDrawing(e.touches[0]);
        });
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            this.draw(e.touches[0]);
        });
        canvas.addEventListener('touchend', () => this.stopDrawing());
    }

    /**
     * Initialize drawing canvas when tab is shown
     */
    initDrawingCanvas() {
        this.clearDrawingCanvas();
    }

    /**
     * Start drawing
     */
    startDrawing(e) {
        this.isDrawing = true;
        const rect = e.target.getBoundingClientRect();
        this.lastDrawPos = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    /**
     * Draw on canvas
     */
    draw(e) {
        if (!this.isDrawing) return;

        const ctx = this.contexts.drawCanvas;
        if (!ctx) return;

        const rect = ctx.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Draw white line on black background (MNIST style)
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        ctx.beginPath();
        ctx.moveTo(this.lastDrawPos.x, this.lastDrawPos.y);
        ctx.lineTo(x, y);
        ctx.stroke();

        this.lastDrawPos = { x, y };

        // Live prediction on draw
        this.predictDrawing();
    }

    /**
     * Stop drawing
     */
    stopDrawing() {
        this.isDrawing = false;
    }

    /**
     * Clear the drawing canvas
     */
    clearDrawingCanvas() {
        const ctx = this.contexts.drawCanvas;
        if (!ctx) return;

        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        // Clear processed preview
        const processedCtx = this.contexts.drawProcessed;
        if (processedCtx) {
            processedCtx.fillStyle = 'black';
            processedCtx.fillRect(0, 0, processedCtx.canvas.width, processedCtx.canvas.height);
        }

        // Clear prediction
        const predContainer = document.getElementById('explainer-draw-prediction');
        if (predContainer) {
            predContainer.textContent = '';
        }
    }

    /**
     * Predict from drawn image
     */
    async predictDrawing() {
        const ctx = this.contexts.drawCanvas;
        if (!ctx || !this.modelBuilder.model) return;

        // Get canvas data
        const canvas = ctx.canvas;
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        // Process to 28x28 grayscale
        const config = this.datasetLoader.getConfig();
        const targetSize = config.imageSize;

        // Create temp canvas for resizing
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = targetSize;
        tempCanvas.height = targetSize;
        const tempCtx = tempCanvas.getContext('2d');

        // Center and resize the drawn content
        tempCtx.fillStyle = 'black';
        tempCtx.fillRect(0, 0, targetSize, targetSize);

        // Find bounding box of drawn content
        const bounds = this.getDrawingBounds(imageData);
        if (!bounds) return;  // Nothing drawn

        // Calculate centered, scaled position
        const padding = 4;
        const maxSize = targetSize - padding * 2;
        const scale = Math.min(maxSize / bounds.width, maxSize / bounds.height);
        const newWidth = bounds.width * scale;
        const newHeight = bounds.height * scale;
        const dx = (targetSize - newWidth) / 2;
        const dy = (targetSize - newHeight) / 2;

        tempCtx.drawImage(
            canvas,
            bounds.x, bounds.y, bounds.width, bounds.height,
            dx, dy, newWidth, newHeight
        );

        // Show processed image
        const processedCtx = this.contexts.drawProcessed;
        if (processedCtx) {
            processedCtx.imageSmoothingEnabled = false;
            processedCtx.fillStyle = 'black';
            processedCtx.fillRect(0, 0, processedCtx.canvas.width, processedCtx.canvas.height);
            processedCtx.drawImage(tempCanvas, 0, 0, processedCtx.canvas.width, processedCtx.canvas.height);
        }

        // Convert to tensor
        const processedData = tempCtx.getImageData(0, 0, targetSize, targetSize);
        const normalized = new Float32Array(targetSize * targetSize);

        for (let i = 0; i < targetSize * targetSize; i++) {
            // Use red channel (same as green/blue for grayscale)
            normalized[i] = processedData.data[i * 4] / 255.0;
        }

        // Create tensor and predict
        const inputTensor = tf.tensor4d(normalized, [1, targetSize, targetSize, config.channels]);

        try {
            const pred = this.modelBuilder.predict(inputTensor);
            const probs = await pred.data();
            pred.dispose();

            this.renderDrawingPrediction(probs);
        } finally {
            inputTensor.dispose();
        }
    }

    /**
     * Get bounding box of drawn content
     */
    getDrawingBounds(imageData) {
        const { data, width, height } = imageData;
        let minX = width, minY = height, maxX = 0, maxY = 0;
        let hasContent = false;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                // Check if pixel is bright enough
                if (data[idx] > 10) {
                    hasContent = true;
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }

        if (!hasContent) return null;

        return {
            x: minX,
            y: minY,
            width: maxX - minX + 1,
            height: maxY - minY + 1
        };
    }

    /**
     * Render prediction for drawn image
     */
    renderDrawingPrediction(probs) {
        const container = document.getElementById('explainer-draw-prediction');
        if (!container) return;

        const config = this.datasetLoader.getConfig();
        container.textContent = '';

        // Find max
        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) {
            if (probs[i] > probs[maxIdx]) maxIdx = i;
        }

        // Sort by probability
        const sorted = Array.from(probs)
            .map((p, i) => ({ prob: p, idx: i }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 5);

        sorted.forEach(({ prob, idx }) => {
            const isTop = idx === maxIdx;

            const bar = document.createElement('div');
            bar.className = 'explainer-pred-bar';

            const label = document.createElement('span');
            label.className = 'explainer-pred-bar__label';
            label.textContent = config.classNames[idx] || `Class ${idx}`;

            const track = document.createElement('div');
            track.className = 'explainer-pred-bar__track';

            const fill = document.createElement('div');
            fill.className = 'explainer-pred-bar__fill';
            if (isTop) fill.classList.add('explainer-pred-bar__fill--correct');
            fill.style.width = `${prob * 100}%`;

            const value = document.createElement('span');
            value.className = 'explainer-pred-bar__value';
            value.textContent = `${(prob * 100).toFixed(1)}%`;

            track.appendChild(fill);
            bar.appendChild(label);
            bar.appendChild(track);
            bar.appendChild(value);
            container.appendChild(bar);
        });

        // Show predicted class prominently
        const result = document.createElement('div');
        result.className = 'explainer-pred-summary explainer-pred-summary--correct';
        result.textContent = `Prediction: ${config.classNames[maxIdx] || maxIdx} (${(probs[maxIdx] * 100).toFixed(1)}%)`;
        container.appendChild(result);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FLATTEN VISUALIZATION
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Render flatten layer visualization
     */
    async renderFlattenVisualization() {
        if (!this.currentSample || !this.modelBuilder.model) return;

        // Find flatten layer index
        let flattenIdx = -1;
        for (let i = 0; i < this.modelBuilder.layers.length; i++) {
            if (this.modelBuilder.layers[i].type === 'flatten') {
                flattenIdx = i;
                break;
            }
        }

        if (flattenIdx < 0) {
            this.showFlattenError('No Flatten layer in model');
            return;
        }

        // Get activation before flatten (last conv/pool layer)
        const beforeIdx = flattenIdx - 1;
        const beforeActivation = this.activations[beforeIdx];
        const afterActivation = this.activations[flattenIdx];

        if (!beforeActivation || !afterActivation) {
            this.showFlattenError('Could not get activation data');
            return;
        }

        await this.renderFlattenBefore(beforeActivation);
        await this.renderFlattenAfter(afterActivation);
        this.updateFlattenInfo(beforeActivation, afterActivation);
    }

    /**
     * Show error message in flatten panel
     */
    showFlattenError(message) {
        const ctx = this.contexts.flattenBefore;
        if (ctx) {
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            ctx.fillStyle = '#64748b';
            ctx.font = '14px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(message, ctx.canvas.width / 2, ctx.canvas.height / 2);
        }
    }

    /**
     * Render 2D feature maps before flatten
     */
    async renderFlattenBefore(activation) {
        const ctx = this.contexts.flattenBefore;
        if (!ctx) return;

        const canvas = ctx.canvas;
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const shape = activation.shape;
        if (shape.length !== 4) return;

        const [, h, w, channels] = shape;
        const data = await activation.data();

        // Show a subset of channels (4x4 grid = 16 channels)
        const maxChannels = Math.min(16, channels);
        const cols = 4;
        const rows = Math.ceil(maxChannels / cols);
        const padding = 4;
        const mapSize = (canvas.width - padding * (cols + 1)) / cols;

        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';

        for (let c = 0; c < maxChannels; c++) {
            const col = c % cols;
            const row = Math.floor(c / cols);
            const x = padding + col * (mapSize + padding);
            const y = padding + row * (mapSize + padding);

            // Extract channel data
            const channelData = new Float32Array(h * w);
            for (let i = 0; i < h; i++) {
                for (let j = 0; j < w; j++) {
                    const idx = i * w * channels + j * channels + c;
                    channelData[i * w + j] = data[idx];
                }
            }

            let minVal = Infinity, maxVal = -Infinity;
            for (let i = 0; i < channelData.length; i++) {
                minVal = Math.min(minVal, channelData[i]);
                maxVal = Math.max(maxVal, channelData[i]);
            }

            this.drawHeatmap(ctx, channelData, w, h, x, y, mapSize, minVal, maxVal - minVal, true);
        }

        // Title
        ctx.fillStyle = '#64748b';
        ctx.font = '10px Inter, sans-serif';
        ctx.fillText(`${channels} channels × ${h}×${w}`, canvas.width / 2, canvas.height - 5);
    }

    /**
     * Render 1D vector after flatten
     */
    async renderFlattenAfter(activation) {
        const ctx = this.contexts.flattenAfter;
        if (!ctx) return;

        const canvas = ctx.canvas;
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const shape = activation.shape;
        if (shape.length !== 2) return;

        const [, length] = shape;
        const data = await activation.data();

        // Find min/max for normalization
        let minVal = Infinity, maxVal = -Infinity;
        for (let i = 0; i < length; i++) {
            minVal = Math.min(minVal, data[i]);
            maxVal = Math.max(maxVal, data[i]);
        }
        const range = maxVal - minVal || 1;

        // Draw vertical bar of values
        const barWidth = canvas.width - 4;
        const barHeight = canvas.height - 20;
        const elemHeight = Math.max(1, barHeight / Math.min(length, 300));

        for (let i = 0; i < Math.min(length, 300); i++) {
            const val = (data[i] - minVal) / range;
            const [r, g, b] = this.viridisColor(val);

            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(2, 10 + i * elemHeight, barWidth, Math.max(1, elemHeight - 0.5));
        }

        // Border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.strokeRect(2, 10, barWidth, barHeight);

        // Label
        ctx.fillStyle = '#64748b';
        ctx.font = '9px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${length}`, canvas.width / 2, canvas.height - 2);
    }

    /**
     * Update flatten info display
     */
    updateFlattenInfo(beforeActivation, afterActivation) {
        const infoEl = document.getElementById('flatten-info');
        if (!infoEl) return;

        const beforeShape = beforeActivation.shape;
        const afterShape = afterActivation.shape;

        const [, h, w, c] = beforeShape;
        const [, length] = afterShape;

        infoEl.textContent = `[${h}, ${w}, ${c}] → [${length}]  |  ${h} × ${w} × ${c} = ${h * w * c} values`;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SOFTMAX VISUALIZATION
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Render softmax visualization
     */
    async renderSoftmaxVisualization() {
        if (!this.currentSample || !this.modelBuilder.model) return;

        // Find the last dense layer (before softmax) by getting activations
        const lastLayerIdx = this.modelBuilder.layers.length - 1;
        const lastLayer = this.modelBuilder.layers[lastLayerIdx];

        // Get logits (pre-softmax values)
        try {
            const logits = await this.getLogits();
            if (!logits) return;

            const logitsData = await logits.data();
            const config = this.datasetLoader.getConfig();

            // Compute softmax manually for visualization
            const expValues = [];
            let expSum = 0;
            for (let i = 0; i < logitsData.length; i++) {
                const exp = Math.exp(logitsData[i]);
                expValues.push(exp);
                expSum += exp;
            }

            const probs = expValues.map(e => e / expSum);

            // Render both visualizations
            this.renderLogitsChart(logitsData, config);
            this.renderProbsChart(probs, config);
            this.renderSoftmaxDetails(logitsData, expValues, probs, expSum, config);

            logits.dispose();
        } catch (e) {
            console.warn('Could not compute softmax visualization:', e.message);
        }
    }

    /**
     * Get raw logits (pre-softmax output)
     */
    async getLogits() {
        if (!this.currentSample || !this.modelBuilder.model) return null;

        // Find the last layer - if it has softmax activation, we need pre-softmax values
        // For simplicity, we'll use a modified forward pass that stops before softmax
        const lastLayerIdx = this.modelBuilder.layers.length - 1;
        const lastLayerConfig = this.modelBuilder.layers[lastLayerIdx];

        // Get the second-to-last dense layer output (before softmax)
        // This is a simplification - ideally we'd create a model without the final activation
        let targetLayerIdx = lastLayerIdx;

        // Try to get pre-softmax values by computing with a linear activation
        return tf.tidy(() => {
            // Get the output before the final activation
            // This is approximate - we use the last intermediate model
            const activation = this.activations[lastLayerIdx];
            if (activation) {
                // For demonstration, we'll use the final activations
                // In a real implementation, you'd want pre-softmax logits
                // We'll simulate logits by taking log of the probabilities
                const probs = activation.squeeze();
                // Simulate logits from probs (inverse of softmax, approximately)
                const logProbs = probs.log();
                const meanLogProb = logProbs.mean();
                return logProbs.sub(meanLogProb);  // Center around 0
            }
            return null;
        });
    }

    /**
     * Render logits bar chart
     */
    renderLogitsChart(logitsData, config) {
        const ctx = this.contexts.softmaxLogits;
        if (!ctx) return;

        const canvas = ctx.canvas;
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const numClasses = logitsData.length;
        const padding = { top: 30, bottom: 40, left: 50, right: 20 };
        const chartWidth = canvas.width - padding.left - padding.right;
        const chartHeight = canvas.height - padding.top - padding.bottom;
        const barWidth = (chartWidth / numClasses) * 0.7;
        const barGap = (chartWidth / numClasses) * 0.3;

        // Find range
        let minVal = Math.min(...logitsData);
        let maxVal = Math.max(...logitsData);
        const absMax = Math.max(Math.abs(minVal), Math.abs(maxVal), 1);

        // Title
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Raw Logits (z)', canvas.width / 2, 18);

        // Zero line
        const zeroY = padding.top + chartHeight / 2;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(padding.left, zeroY);
        ctx.lineTo(canvas.width - padding.right, zeroY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw bars
        for (let i = 0; i < numClasses; i++) {
            const val = logitsData[i];
            const normalizedVal = val / absMax;
            const barHeight = Math.abs(normalizedVal) * (chartHeight / 2);

            const x = padding.left + i * (barWidth + barGap);
            const y = val >= 0 ? zeroY - barHeight : zeroY;

            // Color: warm for positive, cool for negative
            const hue = val >= 0 ? 30 : 220;
            const saturation = Math.min(80, Math.abs(normalizedVal) * 100);
            ctx.fillStyle = `hsl(${hue}, ${saturation}%, 55%)`;

            ctx.fillRect(x, y, barWidth, barHeight);

            // Class label
            ctx.fillStyle = '#64748b';
            ctx.font = '10px Inter, sans-serif';
            ctx.textAlign = 'center';
            const label = config.classNames[i] || i.toString();
            ctx.fillText(label, x + barWidth / 2, canvas.height - padding.bottom + 15);

            // Value label
            ctx.fillStyle = '#94a3b8';
            ctx.font = '9px JetBrains Mono, monospace';
            const valY = val >= 0 ? y - 5 : y + barHeight + 12;
            ctx.fillText(val.toFixed(2), x + barWidth / 2, valY);
        }

        // Y-axis label
        ctx.save();
        ctx.translate(15, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = '#64748b';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Value', 0, 0);
        ctx.restore();
    }

    /**
     * Render probabilities bar chart
     */
    renderProbsChart(probs, config) {
        const ctx = this.contexts.softmaxProbs;
        if (!ctx) return;

        const canvas = ctx.canvas;
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const numClasses = probs.length;
        const padding = { top: 30, bottom: 40, left: 50, right: 20 };
        const chartWidth = canvas.width - padding.left - padding.right;
        const chartHeight = canvas.height - padding.top - padding.bottom;
        const barWidth = (chartWidth / numClasses) * 0.7;
        const barGap = (chartWidth / numClasses) * 0.3;

        // Find max probability
        let maxProb = 0;
        let maxIdx = 0;
        for (let i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIdx = i;
            }
        }

        // Title
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Probabilities (σ)', canvas.width / 2, 18);

        // Draw bars from bottom
        for (let i = 0; i < numClasses; i++) {
            const prob = probs[i];
            const barHeight = prob * chartHeight;

            const x = padding.left + i * (barWidth + barGap);
            const y = padding.top + chartHeight - barHeight;

            // Color based on probability
            const isMax = i === maxIdx;
            if (isMax) {
                ctx.fillStyle = '#22c55e';  // Green for highest
            } else {
                const brightness = 30 + prob * 40;
                ctx.fillStyle = `hsl(220, 70%, ${brightness}%)`;
            }

            ctx.fillRect(x, y, barWidth, barHeight);

            // Class label
            ctx.fillStyle = '#64748b';
            ctx.font = '10px Inter, sans-serif';
            ctx.textAlign = 'center';
            const label = config.classNames[i] || i.toString();
            ctx.fillText(label, x + barWidth / 2, canvas.height - padding.bottom + 15);

            // Value label
            ctx.fillStyle = isMax ? '#22c55e' : '#94a3b8';
            ctx.font = '9px JetBrains Mono, monospace';
            ctx.fillText(`${(prob * 100).toFixed(1)}%`, x + barWidth / 2, y - 5);
        }

        // Y-axis label
        ctx.save();
        ctx.translate(15, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = '#64748b';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Probability', 0, 0);
        ctx.restore();
    }

    /**
     * Render softmax calculation details
     */
    renderSoftmaxDetails(logits, expValues, probs, expSum, config) {
        const container = document.getElementById('softmax-details');
        if (!container) return;

        container.textContent = '';

        // Header row
        const header = document.createElement('div');
        header.className = 'softmax-details__header';

        const headerClass = document.createElement('span');
        headerClass.className = 'softmax-details__class';
        headerClass.textContent = 'Class';

        const headerLogit = document.createElement('span');
        headerLogit.className = 'softmax-details__logit';
        headerLogit.textContent = 'Logit (z)';

        const headerExp = document.createElement('span');
        headerExp.className = 'softmax-details__exp';
        headerExp.textContent = 'exp(z)';

        const headerProb = document.createElement('span');
        headerProb.className = 'softmax-details__prob';
        headerProb.textContent = 'Prob';

        header.appendChild(headerClass);
        header.appendChild(headerLogit);
        header.appendChild(headerExp);
        header.appendChild(headerProb);
        container.appendChild(header);

        // Data rows
        for (let i = 0; i < logits.length; i++) {
            const row = document.createElement('div');
            row.className = 'softmax-details__row';

            const classSpan = document.createElement('span');
            classSpan.className = 'softmax-details__class';
            classSpan.textContent = config.classNames[i] || `${i}`;

            const logitSpan = document.createElement('span');
            logitSpan.className = 'softmax-details__logit';
            logitSpan.textContent = logits[i].toFixed(3);

            const expSpan = document.createElement('span');
            expSpan.className = 'softmax-details__exp';
            expSpan.textContent = expValues[i].toFixed(3);

            const probSpan = document.createElement('span');
            probSpan.className = 'softmax-details__prob';
            probSpan.textContent = `${(probs[i] * 100).toFixed(1)}%`;

            row.appendChild(classSpan);
            row.appendChild(logitSpan);
            row.appendChild(expSpan);
            row.appendChild(probSpan);
            container.appendChild(row);
        }

        // Sum row
        const sumRow = document.createElement('div');
        sumRow.className = 'softmax-details__sum';
        sumRow.textContent = `Σ exp(z) = ${expSum.toFixed(3)}  |  Σ prob = ${probs.reduce((a, b) => a + b, 0).toFixed(3)}`;
        container.appendChild(sumRow);
    }

    /**
     * Dispose resources
     */
    dispose() {
        this.close();
    }
}
