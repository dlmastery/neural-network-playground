/**
 * GNN Model
 * Graph Neural Network implementation with GCN and GAT layers
 * Built from scratch in vanilla JavaScript
 */

/**
 * Layer type configurations
 */
export const GNN_LAYER_TYPES = {
    gcn: {
        name: 'Graph Convolution (GCN)',
        shortName: 'GCN',
        color: '#3b82f6',
        formula: "H' = ReLU(D⁻¹/²AD⁻¹/²HW)"
    },
    gat: {
        name: 'Graph Attention (GAT)',
        shortName: 'GAT',
        color: '#8b5cf6',
        formula: "H' = ReLU(Σ αᵢⱼ Whⱼ)"
    }
};

/**
 * Matrix multiplication helper
 * A: [m, k], B: [k, n] -> C: [m, n]
 */
function matmul(A, m, k, B, n) {
    const C = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
    return C;
}

/**
 * ReLU activation
 */
function relu(x) {
    return x.map(v => Math.max(0, v));
}

/**
 * Softmax over last dimension
 */
function softmax(logits, numNodes, numClasses) {
    const probs = new Float32Array(numNodes * numClasses);

    for (let i = 0; i < numNodes; i++) {
        let maxVal = -Infinity;
        for (let j = 0; j < numClasses; j++) {
            maxVal = Math.max(maxVal, logits[i * numClasses + j]);
        }

        let sum = 0;
        for (let j = 0; j < numClasses; j++) {
            const exp = Math.exp(logits[i * numClasses + j] - maxVal);
            probs[i * numClasses + j] = exp;
            sum += exp;
        }

        for (let j = 0; j < numClasses; j++) {
            probs[i * numClasses + j] /= sum;
        }
    }

    return probs;
}

/**
 * Xavier weight initialization
 */
function xavierInit(rows, cols) {
    const weights = new Float32Array(rows * cols);
    const scale = Math.sqrt(2.0 / (rows + cols));
    for (let i = 0; i < weights.length; i++) {
        weights[i] = (Math.random() * 2 - 1) * scale;
    }
    return weights;
}

/**
 * GNN Model class
 */
export class GNNModel {
    constructor(config) {
        this.inputDim = config.inputDim;
        this.hiddenDim = config.hiddenDim || 16;
        this.outputDim = config.outputDim;
        this.numLayers = config.numLayers || 2;
        this.layerType = config.layerType || 'gcn';
        this.dropout = config.dropout || 0.5;

        // Initialize weights
        this.weights = [];
        this.biases = [];
        this.attentionWeights = []; // For GAT

        this.initWeights();

        // Gradient storage
        this.weightGrads = [];
        this.biasGrads = [];

        // Layer outputs for visualization
        this.layerStates = [];
    }

    /**
     * Initialize layer weights
     */
    initWeights() {
        this.weights = [];
        this.biases = [];
        this.attentionWeights = [];

        const dims = [this.inputDim];
        for (let i = 0; i < this.numLayers - 1; i++) {
            dims.push(this.hiddenDim);
        }
        dims.push(this.outputDim);

        for (let i = 0; i < this.numLayers; i++) {
            const inDim = dims[i];
            const outDim = dims[i + 1];

            this.weights.push(xavierInit(inDim, outDim));
            this.biases.push(new Float32Array(outDim));

            // For GAT: attention weights
            if (this.layerType === 'gat') {
                // a = [a_src, a_dst] for computing attention
                this.attentionWeights.push({
                    a_src: xavierInit(outDim, 1),
                    a_dst: xavierInit(outDim, 1)
                });
            }
        }
    }

    /**
     * Forward pass through the GNN
     */
    forward(nodeFeatures, adjNormalized, adjMatrix, numNodes) {
        this.layerStates = [];
        let H = nodeFeatures;
        const n = numNodes;

        for (let layer = 0; layer < this.numLayers; layer++) {
            const W = this.weights[layer];
            const b = this.biases[layer];
            const inDim = layer === 0 ? this.inputDim : this.hiddenDim;
            const outDim = layer === this.numLayers - 1 ? this.outputDim : this.hiddenDim;

            const layerState = {
                input: new Float32Array(H),
                messages: null,
                preActivation: null,
                output: null,
                attentionWeights: null
            };

            if (this.layerType === 'gcn') {
                // GCN: H' = σ(A_norm @ H @ W + b)

                // Step 1: Transform features H @ W
                const HW = matmul(H, n, inDim, W, outDim);

                // Add bias
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < outDim; j++) {
                        HW[i * outDim + j] += b[j];
                    }
                }

                // Step 2: Aggregate from neighbors (message passing)
                const aggregated = matmul(adjNormalized, n, n, HW, outDim);
                layerState.messages = new Float32Array(aggregated);
                layerState.preActivation = new Float32Array(aggregated);

                // Step 3: Apply activation (except last layer)
                if (layer < this.numLayers - 1) {
                    H = relu(aggregated);
                } else {
                    H = aggregated;
                }
            } else if (this.layerType === 'gat') {
                // GAT: H' = σ(Σ αᵢⱼ W hⱼ)

                // Step 1: Transform features H @ W
                const HW = matmul(H, n, inDim, W, outDim);

                // Add bias
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < outDim; j++) {
                        HW[i * outDim + j] += b[j];
                    }
                }

                // Step 2: Compute attention scores
                const attn = this.attentionWeights[layer];
                const attentionMatrix = new Float32Array(n * n);

                // Compute attention coefficients
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < n; j++) {
                        // Only compute for connected nodes (including self)
                        if (adjMatrix[i * n + j] > 0) {
                            // e_ij = LeakyReLU(a_src @ Wh_i + a_dst @ Wh_j)
                            let score = 0;
                            for (let k = 0; k < outDim; k++) {
                                score += HW[i * outDim + k] * attn.a_src[k];
                                score += HW[j * outDim + k] * attn.a_dst[k];
                            }
                            // LeakyReLU
                            attentionMatrix[i * n + j] = score > 0 ? score : 0.2 * score;
                        } else {
                            attentionMatrix[i * n + j] = -1e9; // Mask
                        }
                    }
                }

                // Softmax per row
                for (let i = 0; i < n; i++) {
                    let maxVal = -Infinity;
                    for (let j = 0; j < n; j++) {
                        if (adjMatrix[i * n + j] > 0) {
                            maxVal = Math.max(maxVal, attentionMatrix[i * n + j]);
                        }
                    }

                    let sum = 0;
                    for (let j = 0; j < n; j++) {
                        if (adjMatrix[i * n + j] > 0) {
                            attentionMatrix[i * n + j] = Math.exp(attentionMatrix[i * n + j] - maxVal);
                            sum += attentionMatrix[i * n + j];
                        } else {
                            attentionMatrix[i * n + j] = 0;
                        }
                    }

                    for (let j = 0; j < n; j++) {
                        attentionMatrix[i * n + j] /= sum || 1;
                    }
                }

                layerState.attentionWeights = new Float32Array(attentionMatrix);

                // Step 3: Aggregate with attention weights
                const aggregated = matmul(attentionMatrix, n, n, HW, outDim);
                layerState.messages = new Float32Array(aggregated);
                layerState.preActivation = new Float32Array(aggregated);

                // Step 4: Apply activation (except last layer)
                if (layer < this.numLayers - 1) {
                    H = relu(aggregated);
                } else {
                    H = aggregated;
                }
            }

            layerState.output = new Float32Array(H);
            this.layerStates.push(layerState);
        }

        // Compute predictions
        const logits = H;
        const probs = softmax(logits, n, this.outputDim);
        const predictions = new Int32Array(n);

        for (let i = 0; i < n; i++) {
            let maxIdx = 0;
            let maxVal = probs[i * this.outputDim];
            for (let j = 1; j < this.outputDim; j++) {
                if (probs[i * this.outputDim + j] > maxVal) {
                    maxVal = probs[i * this.outputDim + j];
                    maxIdx = j;
                }
            }
            predictions[i] = maxIdx;
        }

        return {
            layers: this.layerStates,
            logits,
            probabilities: probs,
            predictions
        };
    }

    /**
     * Compute cross-entropy loss
     */
    computeLoss(probs, labels, numNodes, mask = null) {
        let loss = 0;
        let count = 0;

        for (let i = 0; i < numNodes; i++) {
            if (mask && !mask[i]) continue;

            const trueClass = labels[i];
            const prob = Math.max(probs[i * this.outputDim + trueClass], 1e-10);
            loss -= Math.log(prob);
            count++;
        }

        return count > 0 ? loss / count : 0;
    }

    /**
     * Compute accuracy
     */
    computeAccuracy(predictions, labels, numNodes, mask = null) {
        let correct = 0;
        let count = 0;

        for (let i = 0; i < numNodes; i++) {
            if (mask && !mask[i]) continue;

            if (predictions[i] === labels[i]) {
                correct++;
            }
            count++;
        }

        return count > 0 ? correct / count : 0;
    }

    /**
     * Training step with gradient descent
     * Simplified backprop for educational purposes
     */
    trainStep(nodeFeatures, adjNormalized, adjMatrix, labels, numNodes, learningRate = 0.01, mask = null) {
        // Forward pass
        const result = this.forward(nodeFeatures, adjNormalized, adjMatrix, numNodes);

        // Compute loss
        const loss = this.computeLoss(result.probabilities, labels, numNodes, mask);
        const accuracy = this.computeAccuracy(result.predictions, labels, numNodes, mask);

        // Backward pass (simplified gradient descent)
        // Compute output gradient: dL/dlogits = probs - one_hot(labels)
        const n = numNodes;
        const outDim = this.outputDim;

        // Output layer gradient
        const dLogits = new Float32Array(n * outDim);
        for (let i = 0; i < n; i++) {
            if (mask && !mask[i]) continue;

            for (let j = 0; j < outDim; j++) {
                dLogits[i * outDim + j] = result.probabilities[i * outDim + j];
                if (j === labels[i]) {
                    dLogits[i * outDim + j] -= 1;
                }
            }
        }

        // Backprop through layers (simplified - just update based on input/output)
        let dH = dLogits;

        for (let layer = this.numLayers - 1; layer >= 0; layer--) {
            const inDim = layer === 0 ? this.inputDim : this.hiddenDim;
            const layerOutDim = layer === this.numLayers - 1 ? this.outputDim : this.hiddenDim;
            const input = this.layerStates[layer].input;

            // Gradient for weights: dL/dW = input^T @ dH
            for (let i = 0; i < inDim; i++) {
                for (let j = 0; j < layerOutDim; j++) {
                    let grad = 0;
                    for (let k = 0; k < n; k++) {
                        grad += input[k * inDim + i] * dH[k * layerOutDim + j];
                    }
                    // Update weight with gradient descent
                    this.weights[layer][i * layerOutDim + j] -= learningRate * grad / n;
                }
            }

            // Gradient for bias
            for (let j = 0; j < layerOutDim; j++) {
                let grad = 0;
                for (let k = 0; k < n; k++) {
                    grad += dH[k * layerOutDim + j];
                }
                this.biases[layer][j] -= learningRate * grad / n;
            }

            // Propagate gradient to previous layer (simplified)
            if (layer > 0) {
                const newDH = new Float32Array(n * inDim);
                // dH_prev = dH @ W^T (simplified, ignoring adjacency in backprop for educational purposes)
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < inDim; j++) {
                        let sum = 0;
                        for (let k = 0; k < layerOutDim; k++) {
                            sum += dH[i * layerOutDim + k] * this.weights[layer][j * layerOutDim + k];
                        }
                        // ReLU gradient
                        const preAct = this.layerStates[layer - 1].output[i * inDim + j];
                        newDH[i * inDim + j] = preAct > 0 ? sum : 0;
                    }
                }
                dH = newDH;
            }
        }

        return { loss, accuracy, predictions: result.predictions };
    }

    /**
     * Reset model weights
     */
    reset() {
        this.initWeights();
        this.layerStates = [];
    }

    /**
     * Get model info for display
     */
    getInfo() {
        return {
            type: GNN_LAYER_TYPES[this.layerType],
            inputDim: this.inputDim,
            hiddenDim: this.hiddenDim,
            outputDim: this.outputDim,
            numLayers: this.numLayers,
            totalParams: this.weights.reduce((sum, w) => sum + w.length, 0) +
                        this.biases.reduce((sum, b) => sum + b.length, 0)
        };
    }
}
