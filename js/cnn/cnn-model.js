/**
 * CNN Model Builder
 * Creates and manages TensorFlow.js CNN models with configurable layers
 */

// Available layer types and their default configs
export const LAYER_TYPES = {
    conv2d: {
        name: 'Conv2D',
        color: '#3b82f6',  // Blue
        defaults: { filters: 32, kernelSize: 3, strides: 1, padding: 'same', activation: 'relu' }
    },
    maxPooling2d: {
        name: 'MaxPool2D',
        color: '#22c55e',  // Green
        defaults: { poolSize: 2, strides: 2 }
    },
    avgPooling2d: {
        name: 'AvgPool2D',
        color: '#10b981',  // Emerald
        defaults: { poolSize: 2, strides: 2 }
    },
    batchNormalization: {
        name: 'BatchNorm',
        color: '#8b5cf6',  // Purple
        defaults: {}
    },
    dropout: {
        name: 'Dropout',
        color: '#f59e0b',  // Amber
        defaults: { rate: 0.25 }
    },
    flatten: {
        name: 'Flatten',
        color: '#64748b',  // Slate
        defaults: {}
    },
    dense: {
        name: 'Dense',
        color: '#ef4444',  // Red
        defaults: { units: 128, activation: 'relu' }
    }
};

export const ACTIVATIONS = ['relu', 'sigmoid', 'tanh', 'softmax', 'elu', 'selu', 'linear'];
export const OPTIMIZERS = ['adam', 'sgd', 'rmsprop', 'adagrad'];

export class CNNModelBuilder {
    constructor() {
        this.model = null;
        this.layers = [];
        this.inputShape = [28, 28, 1];
        this.numClasses = 10;
        this.optimizer = 'adam';
        this.learningRate = 0.001;
        this.intermediateModels = [];  // For getting activations at each layer
        this.trainedBatches = 0;  // Track if model has been trained
    }

    /**
     * Check if model has been trained
     */
    isTrained() {
        return this.trainedBatches > 0;
    }

    /**
     * Set input shape based on dataset
     * @param {number[]} shape - [height, width, channels]
     */
    setInputShape(shape) {
        this.inputShape = shape;
    }

    /**
     * Set number of output classes
     * @param {number} numClasses
     */
    setNumClasses(numClasses) {
        this.numClasses = numClasses;
    }

    /**
     * Set optimizer and learning rate
     */
    setOptimizer(optimizer, learningRate = 0.001) {
        this.optimizer = optimizer;
        this.learningRate = learningRate;
    }

    /**
     * Add a layer to the model config
     * @param {string} type - Layer type from LAYER_TYPES
     * @param {object} config - Layer configuration
     */
    addLayer(type, config = {}) {
        const layerType = LAYER_TYPES[type];
        if (!layerType) {
            throw new Error(`Unknown layer type: ${type}`);
        }

        this.layers.push({
            id: Date.now() + Math.random(),
            type,
            config: { ...layerType.defaults, ...config }
        });

        return this.layers[this.layers.length - 1];
    }

    /**
     * Remove layer at index
     * @param {number} index
     */
    removeLayer(index) {
        if (index >= 0 && index < this.layers.length) {
            this.layers.splice(index, 1);
        }
    }

    /**
     * Update layer config
     * @param {number} index
     * @param {object} config
     */
    updateLayer(index, config) {
        if (index >= 0 && index < this.layers.length) {
            this.layers[index].config = { ...this.layers[index].config, ...config };
        }
    }

    /**
     * Move layer to new position
     * @param {number} fromIndex
     * @param {number} toIndex
     */
    moveLayer(fromIndex, toIndex) {
        if (fromIndex >= 0 && fromIndex < this.layers.length &&
            toIndex >= 0 && toIndex < this.layers.length) {
            const [layer] = this.layers.splice(fromIndex, 1);
            this.layers.splice(toIndex, 0, layer);
        }
    }

    /**
     * Get default model configuration
     */
    getDefaultLayers() {
        return [
            { type: 'conv2d', config: { filters: 32, kernelSize: 3, padding: 'same', activation: 'relu' } },
            { type: 'maxPooling2d', config: { poolSize: 2, strides: 2 } },
            { type: 'conv2d', config: { filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' } },
            { type: 'maxPooling2d', config: { poolSize: 2, strides: 2 } },
            { type: 'flatten', config: {} },
            { type: 'dropout', config: { rate: 0.25 } },
            { type: 'dense', config: { units: 128, activation: 'relu' } },
            { type: 'dense', config: { units: this.numClasses, activation: 'softmax' } }
        ];
    }

    /**
     * Load default model configuration
     */
    loadDefault() {
        this.layers = [];
        for (const layerConfig of this.getDefaultLayers()) {
            this.addLayer(layerConfig.type, layerConfig.config);
        }
    }

    /**
     * Build the TensorFlow.js model from layer config
     */
    build() {
        // Reset training state
        this.trainedBatches = 0;

        // Dispose old intermediate models first (they reference the main model's layers)
        this.intermediateModels.forEach(m => {
            if (m) {
                try { m.dispose(); } catch (e) { /* ignore */ }
            }
        });
        this.intermediateModels = [];

        // Dispose old model
        if (this.model) {
            try { this.model.dispose(); } catch (e) { /* ignore */ }
            this.model = null;
        }

        this.model = tf.sequential();

        let isFirstLayer = true;
        let hasFlattened = false;

        for (let i = 0; i < this.layers.length; i++) {
            const { type, config } = this.layers[i];
            let layerConfig = { ...config };

            // Add input shape to first layer
            if (isFirstLayer) {
                layerConfig.inputShape = this.inputShape;
                isFirstLayer = false;
            }

            // Create layer based on type
            let layer;
            switch (type) {
                case 'conv2d':
                    layer = tf.layers.conv2d(layerConfig);
                    break;
                case 'maxPooling2d':
                    layer = tf.layers.maxPooling2d(layerConfig);
                    break;
                case 'avgPooling2d':
                    layer = tf.layers.averagePooling2d(layerConfig);
                    break;
                case 'batchNormalization':
                    layer = tf.layers.batchNormalization(layerConfig);
                    break;
                case 'dropout':
                    layer = tf.layers.dropout(layerConfig);
                    break;
                case 'flatten':
                    layer = tf.layers.flatten(layerConfig);
                    hasFlattened = true;
                    break;
                case 'dense':
                    layer = tf.layers.dense(layerConfig);
                    break;
                default:
                    console.warn(`Unknown layer type: ${type}`);
                    continue;
            }

            this.model.add(layer);
        }

        // Compile the model
        const optimizer = this.createOptimizer();
        this.model.compile({
            optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        // Create intermediate models for activation visualization
        this.createIntermediateModels();

        console.log('Model built:', this.model.summary());
        return this.model;
    }

    /**
     * Create optimizer based on config
     */
    createOptimizer() {
        switch (this.optimizer) {
            case 'adam':
                return tf.train.adam(this.learningRate);
            case 'sgd':
                return tf.train.sgd(this.learningRate);
            case 'rmsprop':
                return tf.train.rmsprop(this.learningRate);
            case 'adagrad':
                return tf.train.adagrad(this.learningRate);
            default:
                return tf.train.adam(this.learningRate);
        }
    }

    /**
     * Create intermediate models for getting activations at each layer
     */
    createIntermediateModels() {
        this.intermediateModels = [];

        if (!this.model || !this.model.layers.length) return;

        const inputLayer = this.model.input;

        for (let i = 0; i < this.model.layers.length; i++) {
            const layer = this.model.layers[i];
            try {
                const intermediateModel = tf.model({
                    inputs: inputLayer,
                    outputs: layer.output
                });
                this.intermediateModels.push(intermediateModel);
            } catch (e) {
                console.warn(`Could not create intermediate model for layer ${i}:`, e.message);
                this.intermediateModels.push(null);
            }
        }
    }

    /**
     * Get activations at a specific layer for an input
     * @param {tf.Tensor} input - Input tensor [batch, h, w, c]
     * @param {number} layerIndex - Layer index
     * @returns {tf.Tensor} Activations
     */
    getActivations(input, layerIndex) {
        if (layerIndex < 0 || layerIndex >= this.intermediateModels.length) {
            return null;
        }

        const model = this.intermediateModels[layerIndex];
        if (!model) return null;

        return model.predict(input);
    }

    /**
     * Get all layer activations for an input
     * @param {tf.Tensor} input - Input tensor
     * @returns {tf.Tensor[]} Array of activations
     */
    getAllActivations(input) {
        return this.intermediateModels.map((model, i) => {
            if (!model) return null;
            return model.predict(input);
        });
    }

    /**
     * Train for one epoch
     * @param {object} batch - {xs, ys} tensors
     * @returns {Promise<{loss: number, accuracy: number}>}
     */
    async trainOnBatch(batch) {
        if (!this.model) {
            throw new Error('Model not built');
        }

        const history = await this.model.fit(batch.xs, batch.ys, {
            epochs: 1,
            batchSize: batch.xs.shape[0],
            verbose: 0
        });

        this.trainedBatches++;  // Track training progress

        return {
            loss: history.history.loss[0],
            accuracy: history.history.acc ? history.history.acc[0] : history.history.accuracy[0]
        };
    }

    /**
     * Evaluate on test data
     * @param {object} batch - {xs, ys} tensors
     * @returns {Promise<{loss: number, accuracy: number}>}
     */
    async evaluate(batch) {
        if (!this.model) {
            throw new Error('Model not built');
        }

        const result = await this.model.evaluate(batch.xs, batch.ys, { verbose: 0 });
        const [loss, accuracy] = await Promise.all([result[0].data(), result[1].data()]);

        return {
            loss: loss[0],
            accuracy: accuracy[0]
        };
    }

    /**
     * Predict on input
     * @param {tf.Tensor} input
     * @returns {tf.Tensor} Predictions
     */
    predict(input) {
        if (!this.model) {
            throw new Error('Model not built');
        }
        return this.model.predict(input);
    }

    /**
     * Get model summary info
     */
    getSummary() {
        if (!this.model) return null;

        let totalParams = 0;
        const layerSummary = this.model.layers.map((layer, i) => {
            const config = layer.getConfig();
            const weights = layer.getWeights();
            let params = 0;
            weights.forEach(w => {
                params += w.size;
            });
            totalParams += params;

            return {
                index: i,
                name: layer.name,
                type: layer.getClassName(),
                outputShape: layer.outputShape,
                params
            };
        });

        return {
            layers: layerSummary,
            totalParams,
            inputShape: this.inputShape,
            outputShape: this.model.outputShape
        };
    }

    /**
     * Get weights of a specific layer (for filter visualization)
     * @param {number} layerIndex
     * @returns {tf.Tensor[]} Weights tensors
     */
    getLayerWeights(layerIndex) {
        if (!this.model || layerIndex >= this.model.layers.length) {
            return null;
        }
        return this.model.layers[layerIndex].getWeights();
    }

    /**
     * Dispose model and free memory
     */
    dispose() {
        // Dispose intermediate models first
        this.intermediateModels.forEach(m => {
            if (m) {
                try { m.dispose(); } catch (e) { /* ignore */ }
            }
        });
        this.intermediateModels = [];

        // Then dispose main model
        if (this.model) {
            try { this.model.dispose(); } catch (e) { /* ignore */ }
            this.model = null;
        }
    }
}
