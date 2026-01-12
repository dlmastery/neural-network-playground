/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Neural Network - Composing Layers into a Trainable Model
 * ═══════════════════════════════════════════════════════════════════════════
 */

import { Tensor, Losses } from './tensor.js';
import { DenseLayer } from './layer.js';

/**
 * Sequential Neural Network
 * Layers are stacked sequentially, output of one feeds into the next
 */
export class NeuralNetwork {
    constructor() {
        this.layers = [];
        this.loss = Losses.binaryCrossEntropy;
        this.lossName = 'binaryCrossEntropy';

        // Training state
        this.epoch = 0;
        this.trainingLoss = 0;
        this.trainingAccuracy = 0;
        this.lossHistory = [];
        this.accuracyHistory = [];

        // Callbacks for visualization
        this.onForwardPass = null;
        this.onBackwardPass = null;
        this.onWeightUpdate = null;
    }

    /**
     * Add a layer to the network
     */
    addLayer(layer) {
        this.layers.push(layer);
        return this;
    }

    /**
     * Build network from architecture specification
     * @param {number[]} architecture - Array of layer sizes [input, hidden..., output]
     * @param {string} activation - Activation function for hidden layers
     * @param {string} outputActivation - Activation function for output layer
     */
    static create(architecture, activation = 'tanh', outputActivation = 'sigmoid') {
        const network = new NeuralNetwork();

        for (let i = 0; i < architecture.length - 1; i++) {
            const isOutputLayer = i === architecture.length - 2;
            const act = isOutputLayer ? outputActivation : activation;

            network.addLayer(new DenseLayer(
                architecture[i],
                architecture[i + 1],
                act
            ));
        }

        return network;
    }

    /**
     * Forward pass through all layers
     */
    forward(input) {
        let output = input;

        for (let i = 0; i < this.layers.length; i++) {
            output = this.layers[i].forward(output);

            if (this.onForwardPass) {
                this.onForwardPass(i, this.layers[i], output);
            }
        }

        return output;
    }

    /**
     * Backward pass through all layers
     */
    backward(gradOutput) {
        let grad = gradOutput;

        for (let i = this.layers.length - 1; i >= 0; i--) {
            grad = this.layers[i].backward(grad);

            if (this.onBackwardPass) {
                this.onBackwardPass(i, this.layers[i], grad);
            }
        }

        return grad;
    }

    /**
     * Compute loss between predictions and targets
     */
    computeLoss(predictions, targets) {
        return this.loss.forward(predictions, targets);
    }

    /**
     * Compute gradient of loss
     */
    computeLossGradient(predictions, targets) {
        return this.loss.backward(predictions, targets);
    }

    /**
     * Single training step
     * @param {Tensor} X - Input batch
     * @param {Tensor} y - Target batch
     * @param {number} learningRate - Learning rate
     * @param {number} l2Lambda - L2 regularization strength
     * @returns {Object} Training metrics
     */
    trainStep(X, y, learningRate = 0.01, l2Lambda = 0) {
        // Forward pass
        const predictions = this.forward(X);

        // Compute loss
        let loss = this.computeLoss(predictions, y);

        // Add L2 regularization to loss
        if (l2Lambda > 0) {
            for (const layer of this.layers) {
                if (layer.weights) {
                    loss += (l2Lambda / 2) * layer.weights.pow(2).sum();
                }
            }
        }

        // Compute accuracy
        const accuracy = this.computeAccuracy(predictions, y);

        // Backward pass
        const gradLoss = this.computeLossGradient(predictions, y);
        this.backward(gradLoss);

        // Update weights using SGD
        for (const layer of this.layers) {
            if (layer.dWeights && layer.dBias) {
                // Add L2 regularization gradient
                if (l2Lambda > 0) {
                    layer.dWeights = layer.dWeights.add(layer.weights.mul(l2Lambda));
                }

                // Update weights
                layer.weights = layer.weights.sub(layer.dWeights.mul(learningRate));
                layer.bias = layer.bias.sub(layer.dBias.mul(learningRate));

                if (this.onWeightUpdate) {
                    this.onWeightUpdate(layer);
                }
            }
        }

        return { loss, accuracy };
    }

    /**
     * Train with Adam optimizer (better convergence)
     */
    trainStepAdam(X, y, learningRate = 0.001, l2Lambda = 0, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
        this.adamT = (this.adamT || 0) + 1;
        const t = this.adamT;

        // Forward pass
        const predictions = this.forward(X);

        // Compute loss
        let loss = this.computeLoss(predictions, y);

        // Check for NaN and reset if needed
        if (isNaN(loss) || !isFinite(loss)) {
            this.reset();
            return { loss: 0, accuracy: 0.5 };
        }

        if (l2Lambda > 0) {
            for (const layer of this.layers) {
                if (layer.weights) {
                    loss += (l2Lambda / 2) * layer.weights.pow(2).sum();
                }
            }
        }

        const accuracy = this.computeAccuracy(predictions, y);

        // Backward pass
        const gradLoss = this.computeLossGradient(predictions, y);
        this.backward(gradLoss);

        // Gradient clipping to prevent explosion
        const maxGradNorm = 5.0;
        for (const layer of this.layers) {
            if (layer.dWeights) {
                layer.dWeights = layer.dWeights.clip(-maxGradNorm, maxGradNorm);
            }
            if (layer.dBias) {
                layer.dBias = layer.dBias.clip(-maxGradNorm, maxGradNorm);
            }
        }

        // Adam update for each layer
        for (const layer of this.layers) {
            if (layer.dWeights && layer.dBias) {
                // Add L2 regularization
                if (l2Lambda > 0) {
                    layer.dWeights = layer.dWeights.add(layer.weights.mul(l2Lambda));
                }

                // Update biased first moment estimate (momentum)
                layer.vWeights = layer.vWeights.mul(beta1).add(layer.dWeights.mul(1 - beta1));
                layer.vBias = layer.vBias.mul(beta1).add(layer.dBias.mul(1 - beta1));

                // Update biased second raw moment estimate
                layer.sWeights = layer.sWeights.mul(beta2).add(layer.dWeights.pow(2).mul(1 - beta2));
                layer.sBias = layer.sBias.mul(beta2).add(layer.dBias.pow(2).mul(1 - beta2));

                // Bias correction
                const vWeightsCorrected = layer.vWeights.mul(1 / (1 - Math.pow(beta1, t)));
                const vBiasCorrected = layer.vBias.mul(1 / (1 - Math.pow(beta1, t)));
                const sWeightsCorrected = layer.sWeights.mul(1 / (1 - Math.pow(beta2, t)));
                const sBiasCorrected = layer.sBias.mul(1 / (1 - Math.pow(beta2, t)));

                // Update weights
                layer.weights = layer.weights.sub(
                    vWeightsCorrected.div(sWeightsCorrected.sqrt().add(epsilon)).mul(learningRate)
                );
                layer.bias = layer.bias.sub(
                    vBiasCorrected.div(sBiasCorrected.sqrt().add(epsilon)).mul(learningRate)
                );

                if (this.onWeightUpdate) {
                    this.onWeightUpdate(layer);
                }
            }
        }

        return { loss, accuracy };
    }

    /**
     * Compute accuracy for binary classification
     */
    computeAccuracy(predictions, targets) {
        let correct = 0;
        for (let i = 0; i < predictions.rows; i++) {
            const predicted = predictions.get(i, 0) >= 0.5 ? 1 : 0;
            const actual = targets.get(i, 0) >= 0.5 ? 1 : 0;
            if (predicted === actual) correct++;
        }
        return correct / predictions.rows;
    }

    /**
     * Predict on new data
     */
    predict(X) {
        return this.forward(X);
    }

    /**
     * Get prediction classes (0 or 1)
     */
    predictClasses(X) {
        const predictions = this.forward(X);
        return predictions.map(p => p >= 0.5 ? 1 : 0);
    }

    /**
     * Reset network to initial state
     */
    reset() {
        for (const layer of this.layers) {
            if (layer.reset) layer.reset();
        }
        this.epoch = 0;
        this.adamT = 0;
        this.lossHistory = [];
        this.accuracyHistory = [];
    }

    /**
     * Get total number of parameters
     */
    get paramCount() {
        return this.layers.reduce((sum, layer) => sum + (layer.paramCount || 0), 0);
    }

    /**
     * Get network architecture as array
     */
    get architecture() {
        if (this.layers.length === 0) return [];

        const arch = [this.layers[0].inputSize];
        for (const layer of this.layers) {
            arch.push(layer.outputSize);
        }
        return arch;
    }

    /**
     * Get all weights for visualization
     * Returns array of { layer, fromNeuron, toNeuron, weight }
     */
    getAllWeights() {
        const weights = [];

        for (let l = 0; l < this.layers.length; l++) {
            const layer = this.layers[l];
            if (!layer.weights) continue;

            for (let i = 0; i < layer.inputSize; i++) {
                for (let j = 0; j < layer.outputSize; j++) {
                    weights.push({
                        layerIndex: l,
                        fromNeuron: i,
                        toNeuron: j,
                        weight: layer.weights.get(i, j)
                    });
                }
            }
        }

        return weights;
    }

    /**
     * Get activations for all neurons (for visualization)
     */
    getActivations() {
        const activations = [];

        // Input layer activations (from the last forward pass)
        if (this.layers.length > 0 && this.layers[0].input) {
            const inputActivations = [];
            for (let j = 0; j < this.layers[0].input.cols; j++) {
                // Average activation across batch
                let sum = 0;
                for (let i = 0; i < this.layers[0].input.rows; i++) {
                    sum += this.layers[0].input.get(i, j);
                }
                inputActivations.push(sum / this.layers[0].input.rows);
            }
            activations.push(inputActivations);
        }

        // Hidden and output layer activations
        for (const layer of this.layers) {
            if (!layer.output) continue;

            const layerActivations = [];
            for (let j = 0; j < layer.output.cols; j++) {
                let sum = 0;
                for (let i = 0; i < layer.output.rows; i++) {
                    sum += layer.output.get(i, j);
                }
                layerActivations.push(sum / layer.output.rows);
            }
            activations.push(layerActivations);
        }

        return activations;
    }

    /**
     * Serialize network to JSON
     */
    toJSON() {
        return {
            layers: this.layers.map(l => l.toJSON()),
            lossName: this.lossName,
            epoch: this.epoch,
            lossHistory: this.lossHistory,
            accuracyHistory: this.accuracyHistory
        };
    }

    /**
     * Load network from JSON
     */
    static fromJSON(json) {
        const network = new NeuralNetwork();
        network.layers = json.layers.map(l => DenseLayer.fromJSON(l));
        network.lossName = json.lossName;
        network.loss = Losses[json.lossName] || Losses.binaryCrossEntropy;
        network.epoch = json.epoch || 0;
        network.lossHistory = json.lossHistory || [];
        network.accuracyHistory = json.accuracyHistory || [];
        return network;
    }
}
