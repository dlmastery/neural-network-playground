/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Layer Classes - Building Blocks for Neural Networks
 * ═══════════════════════════════════════════════════════════════════════════
 */

import { Tensor, Activations } from './tensor.js';

/**
 * Dense (Fully Connected) Layer
 * Computes: output = activation(input @ weights + bias)
 */
export class DenseLayer {
    /**
     * @param {number} inputSize - Number of input features
     * @param {number} outputSize - Number of neurons in this layer
     * @param {string} activation - Activation function name
     */
    constructor(inputSize, outputSize, activation = 'relu') {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activationName = activation;

        // Get activation functions
        this.activation = Activations[activation] || Activations.relu;

        // Initialize weights using appropriate initialization
        if (activation === 'relu' || activation === 'leakyRelu') {
            // He initialization for ReLU variants
            this.weights = Tensor.randomHe(inputSize, outputSize);
        } else {
            // Xavier initialization for others
            this.weights = Tensor.random(inputSize, outputSize);
        }

        // Initialize biases to zero
        this.bias = Tensor.zeros(1, outputSize);

        // Cache for backpropagation
        this.input = null;
        this.z = null;  // Pre-activation
        this.output = null;  // Post-activation

        // Gradients
        this.dWeights = null;
        this.dBias = null;

        // For optimizers (momentum, Adam, etc.)
        this.vWeights = Tensor.zeros(inputSize, outputSize);
        this.vBias = Tensor.zeros(1, outputSize);
        this.sWeights = Tensor.zeros(inputSize, outputSize);
        this.sBias = Tensor.zeros(1, outputSize);
    }

    /**
     * Forward pass
     * @param {Tensor} input - Input tensor (batch_size x input_size)
     * @returns {Tensor} Output tensor (batch_size x output_size)
     */
    forward(input) {
        this.input = input;

        // Linear transformation: z = input @ weights + bias
        this.z = input.matmul(this.weights).add(this.bias);

        // Apply activation function
        this.output = this.activation.forward(this.z);

        return this.output;
    }

    /**
     * Backward pass
     * @param {Tensor} gradOutput - Gradient from next layer
     * @returns {Tensor} Gradient to pass to previous layer
     */
    backward(gradOutput) {
        const batchSize = this.input.rows;

        // Gradient through activation function
        const activationGrad = this.activation.backward(this.z);
        const dz = gradOutput.mul(activationGrad);

        // Gradient for weights: input.T @ dz
        this.dWeights = this.input.transpose().matmul(dz);

        // Gradient for bias: sum across batch
        this.dBias = dz.sumAxis(0);

        // Gradient for previous layer: dz @ weights.T
        const gradInput = dz.matmul(this.weights.transpose());

        return gradInput;
    }

    /**
     * Get all weights (including bias) as flat array for visualization
     */
    getWeightsFlat() {
        return Array.from(this.weights.data);
    }

    /**
     * Reset layer parameters
     */
    reset() {
        if (this.activationName === 'relu' || this.activationName === 'leakyRelu') {
            this.weights = Tensor.randomHe(this.inputSize, this.outputSize);
        } else {
            this.weights = Tensor.random(this.inputSize, this.outputSize);
        }
        this.bias = Tensor.zeros(1, this.outputSize);

        // Reset optimizer state
        this.vWeights = Tensor.zeros(this.inputSize, this.outputSize);
        this.vBias = Tensor.zeros(1, this.outputSize);
        this.sWeights = Tensor.zeros(this.inputSize, this.outputSize);
        this.sBias = Tensor.zeros(1, this.outputSize);
    }

    /**
     * Get number of trainable parameters
     */
    get paramCount() {
        return this.inputSize * this.outputSize + this.outputSize;
    }

    /**
     * Serialize layer to JSON
     */
    toJSON() {
        return {
            type: 'dense',
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            activation: this.activationName,
            weights: Array.from(this.weights.data),
            bias: Array.from(this.bias.data)
        };
    }

    /**
     * Load layer from JSON
     */
    static fromJSON(json) {
        const layer = new DenseLayer(json.inputSize, json.outputSize, json.activation);
        layer.weights = new Tensor(json.inputSize, json.outputSize, new Float32Array(json.weights));
        layer.bias = new Tensor(1, json.outputSize, new Float32Array(json.bias));
        return layer;
    }
}

/**
 * Dropout Layer (for regularization during training)
 */
export class DropoutLayer {
    constructor(rate = 0.5) {
        this.rate = rate;
        this.mask = null;
        this.training = true;
    }

    forward(input) {
        if (!this.training || this.rate === 0) {
            return input.clone();
        }

        // Create dropout mask
        this.mask = new Tensor(input.rows, input.cols);
        const scale = 1 / (1 - this.rate);

        for (let i = 0; i < this.mask.size; i++) {
            this.mask.data[i] = Math.random() > this.rate ? scale : 0;
        }

        return input.mul(this.mask);
    }

    backward(gradOutput) {
        if (!this.training || this.rate === 0) {
            return gradOutput.clone();
        }
        return gradOutput.mul(this.mask);
    }

    get paramCount() {
        return 0;
    }
}
