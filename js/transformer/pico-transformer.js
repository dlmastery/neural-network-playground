/**
 * PicoTransformer - A minimal educational transformer implementation
 *
 * This is a simplified, readable implementation of the core transformer
 * concepts for educational purposes. It demonstrates:
 * - Token embeddings
 * - Positional embeddings
 * - Self-attention mechanism
 * - Multi-head attention
 * - Feed-forward networks
 * - Layer normalization
 *
 * Based on the ideas from picoGPT and "Attention Is All You Need"
 */

// Demo vocabulary for visualization (common words)
export const DEMO_VOCAB = [
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to',
    'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'as', 'if', 'then',
    'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'cat', 'bird', 'fish',
    'runs', 'walks', 'sits', 'stands', 'falls', 'flies', 'swims', 'eats', 'sleeps', 'plays',
    'big', 'small', 'fast', 'slow', 'hot', 'cold', 'new', 'old', 'good', 'bad',
    'hello', 'world', 'how', 'what', 'when', 'where', 'why', 'who', 'which', 'that',
    'this', 'these', 'those', 'here', 'there', 'now', 'then', 'today', 'tomorrow', 'yesterday',
    'data', 'visualization', 'empowers', 'users', 'understand', 'complex', 'patterns', 'information', 'analysis', 'insights',
    'machine', 'learning', 'neural', 'network', 'deep', 'artificial', 'intelligence', 'model', 'training', 'prediction',
    'transformer', 'attention', 'embedding', 'token', 'layer', 'output', 'input', 'hidden', 'weight', 'bias',
    'sat', 'on', 'mat', 'warm', 'sunny', 'day', 'night', 'morning', 'evening', 'afternoon',
    ' ', '.', ',', '!', '?', ':', ';', '-', "'", '"',
    'The', 'A', 'An', 'It', 'He', 'She', 'We', 'They', 'This', 'That'
];

// Create reverse lookup
export const VOCAB_TO_ID = {};
DEMO_VOCAB.forEach((word, idx) => {
    VOCAB_TO_ID[word] = idx;
    VOCAB_TO_ID[word.toLowerCase()] = idx;
});

export class PicoTransformer {
    /**
     * Initialize a minimal transformer
     * @param {Object} config - Configuration
     */
    constructor(config = {}) {
        this.vocabSize = config.vocabSize || DEMO_VOCAB.length;
        this.contextLength = config.contextLength || 128;
        this.embeddingDim = config.embeddingDim || 64;  // Small for demo
        this.numHeads = config.numHeads || 4;
        this.numLayers = config.numLayers || 2;
        this.ffnDim = config.ffnDim || 256;

        // Initialize weights (normally these would be trained)
        this.initializeWeights();

        // For visualization callbacks
        this.onAttention = null;
        this.onEmbedding = null;
        this.onFFN = null;
    }

    /**
     * Tokenize text into token IDs using demo vocabulary
     */
    tokenize(text) {
        const tokens = [];
        const tokenIds = [];

        // Simple word-based tokenization
        const words = text.split(/(\s+)/);

        for (const word of words) {
            if (!word) continue;

            // Check if word is in vocabulary
            const id = VOCAB_TO_ID[word] ?? VOCAB_TO_ID[word.toLowerCase()];
            if (id !== undefined) {
                tokens.push(word);
                tokenIds.push(id);
            } else {
                // Unknown word - split into characters or use UNK
                tokens.push(word);
                tokenIds.push(Math.abs(this.hashString(word)) % this.vocabSize);
            }
        }

        return { tokens, tokenIds };
    }

    /**
     * Convert token IDs back to text
     */
    detokenize(tokenIds) {
        return tokenIds.map(id => DEMO_VOCAB[id] || `[${id}]`).join('');
    }

    /**
     * Simple string hash for unknown words
     */
    hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = ((hash << 5) - hash) + str.charCodeAt(i);
            hash = hash & hash;
        }
        return hash;
    }

    /**
     * Initialize random weights for demonstration
     */
    initializeWeights() {
        // Token embeddings: [vocabSize, embeddingDim]
        this.tokenEmbeddings = this.randomMatrix(this.vocabSize, this.embeddingDim, 0.02);

        // Positional embeddings: [contextLength, embeddingDim]
        this.positionEmbeddings = this.createPositionalEncoding(
            this.contextLength,
            this.embeddingDim
        );

        // Transformer layers
        this.layers = [];
        for (let i = 0; i < this.numLayers; i++) {
            this.layers.push(this.createLayer());
        }

        // Output projection to vocab size
        this.outputProjection = this.randomMatrix(this.embeddingDim, this.vocabSize, 0.02);
    }

    /**
     * Create a single transformer layer
     */
    createLayer() {
        const headDim = this.embeddingDim / this.numHeads;

        return {
            // Self-attention weights
            attention: {
                Wq: this.randomMatrix(this.embeddingDim, this.embeddingDim, 0.02),
                Wk: this.randomMatrix(this.embeddingDim, this.embeddingDim, 0.02),
                Wv: this.randomMatrix(this.embeddingDim, this.embeddingDim, 0.02),
                Wo: this.randomMatrix(this.embeddingDim, this.embeddingDim, 0.02)
            },
            // Feed-forward network weights
            ffn: {
                W1: this.randomMatrix(this.embeddingDim, this.ffnDim, 0.02),
                b1: new Float32Array(this.ffnDim),
                W2: this.randomMatrix(this.ffnDim, this.embeddingDim, 0.02),
                b2: new Float32Array(this.embeddingDim)
            },
            // Layer norms
            ln1: { gamma: this.ones(this.embeddingDim), beta: new Float32Array(this.embeddingDim) },
            ln2: { gamma: this.ones(this.embeddingDim), beta: new Float32Array(this.embeddingDim) }
        };
    }

    /**
     * Create sinusoidal positional encoding
     * This is the original encoding from "Attention Is All You Need"
     */
    createPositionalEncoding(maxLen, dim) {
        const pe = new Float32Array(maxLen * dim);

        for (let pos = 0; pos < maxLen; pos++) {
            for (let i = 0; i < dim; i++) {
                const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dim);

                if (i % 2 === 0) {
                    pe[pos * dim + i] = Math.sin(angle);
                } else {
                    pe[pos * dim + i] = Math.cos(angle);
                }
            }
        }

        return pe;
    }

    /**
     * Forward pass through the transformer
     * @param {number[]} tokenIds - Array of token IDs
     * @param {number} temperature - Temperature for softmax (default 1.0)
     * @returns {Object} - Output logits and comprehensive intermediate states
     */
    forward(tokenIds, temperature = 1.0) {
        const seqLen = tokenIds.length;

        // Comprehensive state for visualization
        const state = {
            // Input
            tokenIds: [...tokenIds],
            tokens: tokenIds.map(id => DEMO_VOCAB[id] || `[${id}]`),

            // Embeddings
            tokenEmbeddings: null,      // [seqLen, embDim]
            posEmbeddings: null,        // [seqLen, embDim]
            combinedEmbeddings: null,   // [seqLen, embDim]

            // Per-layer detailed states
            layers: [],

            // Final output
            logits: null,               // [vocabSize]
            probabilities: null,        // [vocabSize]
            topK: [],                   // Top predictions with words

            // Legacy compatibility
            embeddings: null,
            posEmbeddings: null,
            layerOutputs: [],
            attentionWeights: []
        };

        // Step 1: Token Embeddings
        let x = this.getTokenEmbeddings(tokenIds);
        state.tokenEmbeddings = this.copyArray(x, seqLen, this.embeddingDim);
        state.embeddings = state.tokenEmbeddings; // Legacy

        // Step 2: Positional Embeddings
        const posEmb = this.getPositionalEmbeddings(seqLen);
        state.posEmbeddings = this.copyArray(posEmb, seqLen, this.embeddingDim);

        // Step 3: Combined Embeddings
        x = this.addArrays(x, posEmb, seqLen * this.embeddingDim);
        state.combinedEmbeddings = this.copyArray(x, seqLen, this.embeddingDim);

        if (this.onEmbedding) {
            this.onEmbedding(state);
        }

        // Step 4: Transformer Layers
        for (let layerIdx = 0; layerIdx < this.layers.length; layerIdx++) {
            const layer = this.layers[layerIdx];

            // Layer state with full details
            const layerState = {
                layerIdx,
                // Q, K, V matrices
                Q: null,
                K: null,
                V: null,
                // Attention
                attentionScores: [],    // Pre-softmax [numHeads, seqLen, seqLen]
                attentionWeights: [],   // Post-softmax [numHeads, seqLen, seqLen]
                attentionOutput: null,  // [seqLen, embDim]
                // FFN
                ffnInput: null,
                ffnHidden: null,        // [seqLen, ffnDim]
                ffnOutput: null,        // [seqLen, embDim]
                // Layer output
                output: null            // [seqLen, embDim]
            };

            // Layer Norm 1
            const normed1 = this.layerNorm(x, seqLen, layer.ln1);

            // Multi-Head Self-Attention with full state capture
            const attnResult = this.multiHeadAttentionFull(
                normed1, seqLen, layer.attention
            );

            layerState.Q = attnResult.Q;
            layerState.K = attnResult.K;
            layerState.V = attnResult.V;
            layerState.attentionScores = attnResult.scores;
            layerState.attentionWeights = attnResult.weights;
            layerState.attentionOutput = this.copyArray(attnResult.output, seqLen, this.embeddingDim);

            // Legacy compatibility
            state.attentionWeights.push(attnResult.weights);

            if (this.onAttention) {
                this.onAttention(layerIdx, attnResult.weights, tokenIds);
            }

            // Residual connection after attention
            x = this.addArrays(x, attnResult.output, seqLen * this.embeddingDim);

            // Layer Norm 2
            const normed2 = this.layerNorm(x, seqLen, layer.ln2);
            layerState.ffnInput = this.copyArray(normed2, seqLen, this.embeddingDim);

            // Feed-Forward Network with hidden state capture
            const ffnResult = this.feedForwardFull(normed2, seqLen, layer.ffn);
            layerState.ffnHidden = ffnResult.hidden;
            layerState.ffnOutput = this.copyArray(ffnResult.output, seqLen, this.embeddingDim);

            if (this.onFFN) {
                this.onFFN(layerIdx, ffnResult.output);
            }

            // Residual connection after FFN
            x = this.addArrays(x, ffnResult.output, seqLen * this.embeddingDim);
            layerState.output = this.copyArray(x, seqLen, this.embeddingDim);

            state.layers.push(layerState);
            state.layerOutputs.push(layerState); // Legacy
        }

        // Step 5: Output Projection
        const lastTokenEmb = x.slice((seqLen - 1) * this.embeddingDim, seqLen * this.embeddingDim);
        const logits = this.matmul1D(lastTokenEmb, this.outputProjection, this.embeddingDim, this.vocabSize);
        state.logits = Array.from(logits);

        // Step 6: Softmax with temperature
        const probs = this.softmaxWithTemperature(logits, temperature);
        state.probabilities = Array.from(probs);

        // Step 7: Top-K predictions
        state.topK = this.getTopK(probs, 10);

        return state;
    }

    /**
     * Get top-K predictions from probability distribution
     */
    getTopK(probs, k = 5) {
        const indexed = Array.from(probs).map((p, i) => ({ id: i, prob: p }));
        indexed.sort((a, b) => b.prob - a.prob);

        return indexed.slice(0, k).map(item => ({
            id: item.id,
            token: DEMO_VOCAB[item.id] || `[${item.id}]`,
            probability: item.prob,
            percentage: (item.prob * 100).toFixed(1) + '%'
        }));
    }

    /**
     * Softmax with temperature
     */
    softmaxWithTemperature(logits, temperature = 1.0) {
        const scaled = new Float32Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
            scaled[i] = logits[i] / temperature;
        }

        let max = -Infinity;
        for (let i = 0; i < scaled.length; i++) {
            if (scaled[i] > max) max = scaled[i];
        }

        let sum = 0;
        const probs = new Float32Array(scaled.length);
        for (let i = 0; i < scaled.length; i++) {
            probs[i] = Math.exp(scaled[i] - max);
            sum += probs[i];
        }

        for (let i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }

        return probs;
    }

    /**
     * Multi-Head Self-Attention with full state capture for visualization
     */
    multiHeadAttentionFull(x, seqLen, weights) {
        const headDim = this.embeddingDim / this.numHeads;

        // Project to Q, K, V
        const Q = this.matmul2D(x, weights.Wq, seqLen, this.embeddingDim, this.embeddingDim);
        const K = this.matmul2D(x, weights.Wk, seqLen, this.embeddingDim, this.embeddingDim);
        const V = this.matmul2D(x, weights.Wv, seqLen, this.embeddingDim, this.embeddingDim);

        // Capture Q, K, V for visualization
        const QCopy = this.copyArray(Q, seqLen, this.embeddingDim);
        const KCopy = this.copyArray(K, seqLen, this.embeddingDim);
        const VCopy = this.copyArray(V, seqLen, this.embeddingDim);

        // Split into heads and compute attention for each
        const allHeadOutputs = new Float32Array(seqLen * this.embeddingDim);
        const allScores = [];  // Pre-softmax
        const allWeights = []; // Post-softmax

        for (let h = 0; h < this.numHeads; h++) {
            // Extract head slice [seqLen, headDim]
            const Qh = this.extractHead(Q, seqLen, h, headDim);
            const Kh = this.extractHead(K, seqLen, h, headDim);
            const Vh = this.extractHead(V, seqLen, h, headDim);

            // Compute attention scores: QK^T / sqrt(d_k)
            const scores = new Float32Array(seqLen * seqLen);
            const scale = 1.0 / Math.sqrt(headDim);

            for (let i = 0; i < seqLen; i++) {
                for (let j = 0; j < seqLen; j++) {
                    let dot = 0;
                    for (let k = 0; k < headDim; k++) {
                        dot += Qh[i * headDim + k] * Kh[j * headDim + k];
                    }
                    scores[i * seqLen + j] = dot * scale;
                }
            }

            // Store pre-softmax scores (before masking for visualization)
            const preMaskScores = new Float32Array(scores);
            allScores.push(this.copyArray(preMaskScores, seqLen, seqLen));

            // Apply causal mask (can only attend to previous positions)
            for (let i = 0; i < seqLen; i++) {
                for (let j = i + 1; j < seqLen; j++) {
                    scores[i * seqLen + j] = -Infinity;
                }
            }

            // Softmax
            const attnWeights = this.softmax2D(scores, seqLen, seqLen);
            allWeights.push(this.copyArray(attnWeights, seqLen, seqLen));

            // Apply attention to values: weights @ V
            const headOut = new Float32Array(seqLen * headDim);
            for (let i = 0; i < seqLen; i++) {
                for (let k = 0; k < headDim; k++) {
                    let sum = 0;
                    for (let j = 0; j < seqLen; j++) {
                        sum += attnWeights[i * seqLen + j] * Vh[j * headDim + k];
                    }
                    headOut[i * headDim + k] = sum;
                }
            }

            // Write head output back
            for (let i = 0; i < seqLen; i++) {
                for (let k = 0; k < headDim; k++) {
                    allHeadOutputs[i * this.embeddingDim + h * headDim + k] = headOut[i * headDim + k];
                }
            }
        }

        // Output projection
        const output = this.matmul2D(allHeadOutputs, weights.Wo, seqLen, this.embeddingDim, this.embeddingDim);

        return {
            Q: QCopy,
            K: KCopy,
            V: VCopy,
            scores: allScores,
            weights: allWeights,
            output
        };
    }

    /**
     * Multi-Head Self-Attention (legacy wrapper)
     */
    multiHeadAttention(x, seqLen, weights) {
        const result = this.multiHeadAttentionFull(x, seqLen, weights);
        // Convert weights back to Float32Array format for legacy code
        const flatWeights = result.weights.map(w => {
            const flat = new Float32Array(w.length * w[0].length);
            for (let i = 0; i < w.length; i++) {
                for (let j = 0; j < w[i].length; j++) {
                    flat[i * w[i].length + j] = w[i][j];
                }
            }
            return flat;
        });
        return { output: result.output, weights: flatWeights };
    }

    /**
     * Feed-Forward Network with full state capture
     * FFN(x) = GELU(xW1 + b1)W2 + b2
     */
    feedForwardFull(x, seqLen, weights) {
        // First linear layer + GELU
        const hidden = new Float32Array(seqLen * this.ffnDim);

        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < this.ffnDim; j++) {
                let sum = weights.b1[j];
                for (let k = 0; k < this.embeddingDim; k++) {
                    sum += x[i * this.embeddingDim + k] * weights.W1[k * this.ffnDim + j];
                }
                hidden[i * this.ffnDim + j] = this.gelu(sum);
            }
        }

        // Second linear layer
        const output = new Float32Array(seqLen * this.embeddingDim);

        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < this.embeddingDim; j++) {
                let sum = weights.b2[j];
                for (let k = 0; k < this.ffnDim; k++) {
                    sum += hidden[i * this.ffnDim + k] * weights.W2[k * this.embeddingDim + j];
                }
                output[i * this.embeddingDim + j] = sum;
            }
        }

        return {
            hidden: this.copyArray(hidden, seqLen, this.ffnDim),
            output
        };
    }

    /**
     * Feed-Forward Network (legacy wrapper)
     */
    feedForward(x, seqLen, weights) {
        return this.feedForwardFull(x, seqLen, weights).output;
    }

    /**
     * Layer Normalization
     * Normalizes across the embedding dimension
     */
    layerNorm(x, seqLen, params) {
        const output = new Float32Array(seqLen * this.embeddingDim);
        const eps = 1e-5;

        for (let i = 0; i < seqLen; i++) {
            // Compute mean
            let mean = 0;
            for (let j = 0; j < this.embeddingDim; j++) {
                mean += x[i * this.embeddingDim + j];
            }
            mean /= this.embeddingDim;

            // Compute variance
            let variance = 0;
            for (let j = 0; j < this.embeddingDim; j++) {
                const diff = x[i * this.embeddingDim + j] - mean;
                variance += diff * diff;
            }
            variance /= this.embeddingDim;

            // Normalize and scale
            const std = Math.sqrt(variance + eps);
            for (let j = 0; j < this.embeddingDim; j++) {
                const normalized = (x[i * this.embeddingDim + j] - mean) / std;
                output[i * this.embeddingDim + j] = params.gamma[j] * normalized + params.beta[j];
            }
        }

        return output;
    }

    // ==================== Helper Functions ====================

    /**
     * Get token embeddings for a sequence
     */
    getTokenEmbeddings(tokenIds) {
        const seqLen = tokenIds.length;
        const embeddings = new Float32Array(seqLen * this.embeddingDim);

        for (let i = 0; i < seqLen; i++) {
            const tokenId = tokenIds[i] % 1000; // Mod to fit our small embedding table
            for (let j = 0; j < this.embeddingDim; j++) {
                embeddings[i * this.embeddingDim + j] = this.tokenEmbeddings[tokenId * this.embeddingDim + j];
            }
        }

        return embeddings;
    }

    /**
     * Get positional embeddings for a sequence
     */
    getPositionalEmbeddings(seqLen) {
        return this.positionEmbeddings.slice(0, seqLen * this.embeddingDim);
    }

    /**
     * GELU activation function
     */
    gelu(x) {
        return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
    }

    /**
     * Softmax over rows
     */
    softmax2D(x, rows, cols) {
        const output = new Float32Array(rows * cols);

        for (let i = 0; i < rows; i++) {
            // Find max for numerical stability
            let max = -Infinity;
            for (let j = 0; j < cols; j++) {
                const val = x[i * cols + j];
                if (val > max) max = val;
            }

            // Compute exp and sum
            let sum = 0;
            for (let j = 0; j < cols; j++) {
                const exp = Math.exp(x[i * cols + j] - max);
                output[i * cols + j] = exp;
                sum += exp;
            }

            // Normalize
            for (let j = 0; j < cols; j++) {
                output[i * cols + j] /= sum;
            }
        }

        return output;
    }

    /**
     * Matrix multiplication: [M, K] @ [K, N] -> [M, N]
     */
    matmul2D(a, b, M, K, N) {
        const output = new Float32Array(M * N);

        for (let i = 0; i < M; i++) {
            for (let j = 0; j < N; j++) {
                let sum = 0;
                for (let k = 0; k < K; k++) {
                    sum += a[i * K + k] * b[k * N + j];
                }
                output[i * N + j] = sum;
            }
        }

        return output;
    }

    /**
     * Vector-matrix multiplication: [K] @ [K, N] -> [N]
     */
    matmul1D(a, b, K, N) {
        const output = new Float32Array(N);

        for (let j = 0; j < N; j++) {
            let sum = 0;
            for (let k = 0; k < K; k++) {
                sum += a[k] * b[k * N + j];
            }
            output[j] = sum;
        }

        return output;
    }

    /**
     * Extract a single attention head's slice
     */
    extractHead(x, seqLen, headIdx, headDim) {
        const output = new Float32Array(seqLen * headDim);
        const embDim = this.embeddingDim;

        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < headDim; j++) {
                output[i * headDim + j] = x[i * embDim + headIdx * headDim + j];
            }
        }

        return output;
    }

    /**
     * Element-wise array addition
     */
    addArrays(a, b, length) {
        const output = new Float32Array(length);
        for (let i = 0; i < length; i++) {
            output[i] = a[i] + b[i];
        }
        return output;
    }

    /**
     * Copy array section to 2D format
     */
    copyArray(arr, rows, cols) {
        const copy = [];
        for (let i = 0; i < rows; i++) {
            copy.push(Array.from(arr.slice(i * cols, (i + 1) * cols)));
        }
        return copy;
    }

    /**
     * Create random matrix with given scale
     */
    randomMatrix(rows, cols, scale) {
        const arr = new Float32Array(rows * cols);
        for (let i = 0; i < arr.length; i++) {
            arr[i] = (Math.random() * 2 - 1) * scale;
        }
        return arr;
    }

    /**
     * Create array of ones
     */
    ones(length) {
        const arr = new Float32Array(length);
        arr.fill(1);
        return arr;
    }

    /**
     * Sample next token from logits
     */
    sampleToken(logits, temperature = 1.0) {
        // Apply temperature
        const scaled = new Float32Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
            scaled[i] = logits[i] / temperature;
        }

        // Softmax
        let max = -Infinity;
        for (let i = 0; i < scaled.length; i++) {
            if (scaled[i] > max) max = scaled[i];
        }

        let sum = 0;
        const probs = new Float32Array(scaled.length);
        for (let i = 0; i < scaled.length; i++) {
            probs[i] = Math.exp(scaled[i] - max);
            sum += probs[i];
        }

        for (let i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }

        // Sample
        const r = Math.random();
        let cumsum = 0;
        for (let i = 0; i < probs.length; i++) {
            cumsum += probs[i];
            if (r < cumsum) return i;
        }

        return probs.length - 1;
    }

    /**
     * Generate text token by token
     */
    *generate(tokenIds, maxTokens = 10, temperature = 1.0) {
        const tokens = [...tokenIds];

        for (let step = 0; step < maxTokens; step++) {
            // Forward pass
            const state = this.forward(tokens);

            // Sample next token
            const nextToken = this.sampleToken(state.logits, temperature);
            tokens.push(nextToken);

            yield {
                step,
                token: nextToken,
                tokens: [...tokens],
                state
            };
        }
    }
}

/**
 * Educational explanation of transformer concepts
 */
export const TransformerConcepts = {
    selfAttention: `
## Self-Attention Mechanism

Self-attention allows each token to "look at" all other tokens in the sequence.

**The Formula:**
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

**Step by step:**
1. **Query (Q)**: "What am I looking for?"
2. **Key (K)**: "What do I contain?"
3. **Value (V)**: "What information do I provide?"

4. QK^T computes compatibility scores
5. Scale by √d_k for stable gradients
6. Softmax normalizes to probabilities
7. Multiply by V to get weighted values
`,

    positionalEncoding: `
## Positional Encoding

Transformers process all tokens in parallel, so they need position information.

**Sinusoidal Encoding:**
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

**Why this works:**
- Each position gets a unique encoding
- Relative positions can be computed via linear transformation
- Can extrapolate to longer sequences
`,

    multiHead: `
## Multi-Head Attention

Instead of one attention mechanism, use several in parallel.

**Why multiple heads?**
- Different heads can learn different relationships
- Head 1: syntax (subject-verb)
- Head 2: semantics (meaning)
- Head 3: coreference (pronouns → nouns)

**How it works:**
1. Split embedding into h parts
2. Each head does attention on its part
3. Concatenate results
4. Project back to embedding size
`,

    ffn: `
## Feed-Forward Network

After attention, each position goes through the same FFN.

**Structure:**
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂

**Purpose:**
- Adds non-linearity
- Processes information per-position
- Often 4× the embedding dimension
- Acts as a "memory" storing knowledge
`,

    layerNorm: `
## Layer Normalization

Normalizes activations for stable training.

**Formula:**
LayerNorm(x) = γ × (x - μ) / σ + β

**Where:**
- μ = mean across features
- σ = standard deviation
- γ, β = learned scale and shift

**Benefits:**
- Stabilizes gradients
- Allows deeper networks
- Applied before attention and FFN (Pre-LN)
`
};
