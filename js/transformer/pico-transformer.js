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

export class PicoTransformer {
    /**
     * Initialize a minimal transformer
     * @param {Object} config - Configuration
     */
    constructor(config = {}) {
        this.vocabSize = config.vocabSize || 50257;     // GPT-2 vocab size
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
     * Initialize random weights for demonstration
     */
    initializeWeights() {
        // Token embeddings: [vocabSize, embeddingDim]
        // For demo, we use a small subset
        this.tokenEmbeddings = this.randomMatrix(1000, this.embeddingDim, 0.02);

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

        // Output projection
        this.outputProjection = this.randomMatrix(this.embeddingDim, 1000, 0.02);
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
     * @returns {Object} - Output logits and intermediate states
     */
    forward(tokenIds) {
        const seqLen = tokenIds.length;
        const intermediateStates = {
            embeddings: null,
            posEmbeddings: null,
            combinedEmbeddings: null,
            layerOutputs: [],
            attentionWeights: [],
            logits: null
        };

        // Step 1: Token Embeddings
        // Look up embedding vector for each token
        let x = this.getTokenEmbeddings(tokenIds);
        intermediateStates.embeddings = this.copyArray(x, seqLen, this.embeddingDim);

        // Step 2: Add Positional Embeddings
        // Each position gets a unique encoding so the model knows word order
        const posEmb = this.getPositionalEmbeddings(seqLen);
        intermediateStates.posEmbeddings = this.copyArray(posEmb, seqLen, this.embeddingDim);

        x = this.addArrays(x, posEmb, seqLen * this.embeddingDim);
        intermediateStates.combinedEmbeddings = this.copyArray(x, seqLen, this.embeddingDim);

        if (this.onEmbedding) {
            this.onEmbedding(intermediateStates);
        }

        // Step 3: Transformer Layers
        for (let layerIdx = 0; layerIdx < this.layers.length; layerIdx++) {
            const layer = this.layers[layerIdx];
            const layerState = {};

            // 3a: Layer Norm 1
            const normed1 = this.layerNorm(x, seqLen, layer.ln1);

            // 3b: Multi-Head Self-Attention
            const { output: attnOut, weights } = this.multiHeadAttention(
                normed1, seqLen, layer.attention
            );
            layerState.attentionWeights = weights;
            intermediateStates.attentionWeights.push(weights);

            if (this.onAttention) {
                this.onAttention(layerIdx, weights, tokenIds);
            }

            // 3c: Residual connection
            x = this.addArrays(x, attnOut, seqLen * this.embeddingDim);

            // 3d: Layer Norm 2
            const normed2 = this.layerNorm(x, seqLen, layer.ln2);

            // 3e: Feed-Forward Network
            const ffnOut = this.feedForward(normed2, seqLen, layer.ffn);

            if (this.onFFN) {
                this.onFFN(layerIdx, ffnOut);
            }

            // 3f: Residual connection
            x = this.addArrays(x, ffnOut, seqLen * this.embeddingDim);

            layerState.output = this.copyArray(x, seqLen, this.embeddingDim);
            intermediateStates.layerOutputs.push(layerState);
        }

        // Step 4: Output Projection (only last token for generation)
        const lastTokenEmb = x.slice((seqLen - 1) * this.embeddingDim, seqLen * this.embeddingDim);
        const logits = this.matmul1D(lastTokenEmb, this.outputProjection, this.embeddingDim, 1000);
        intermediateStates.logits = logits;

        return intermediateStates;
    }

    /**
     * Multi-Head Self-Attention
     *
     * This is the core of the transformer:
     * 1. Project input to Q, K, V
     * 2. Split into multiple heads
     * 3. Compute attention: softmax(QK^T / sqrt(d_k)) * V
     * 4. Concatenate heads and project
     */
    multiHeadAttention(x, seqLen, weights) {
        const headDim = this.embeddingDim / this.numHeads;

        // Project to Q, K, V
        const Q = this.matmul2D(x, weights.Wq, seqLen, this.embeddingDim, this.embeddingDim);
        const K = this.matmul2D(x, weights.Wk, seqLen, this.embeddingDim, this.embeddingDim);
        const V = this.matmul2D(x, weights.Wv, seqLen, this.embeddingDim, this.embeddingDim);

        // Split into heads and compute attention for each
        const allHeadOutputs = new Float32Array(seqLen * this.embeddingDim);
        const allWeights = [];

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

            // Apply causal mask (can only attend to previous positions)
            for (let i = 0; i < seqLen; i++) {
                for (let j = i + 1; j < seqLen; j++) {
                    scores[i * seqLen + j] = -Infinity;
                }
            }

            // Softmax
            const attnWeights = this.softmax2D(scores, seqLen, seqLen);
            allWeights.push(new Float32Array(attnWeights));

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

        return { output, weights: allWeights };
    }

    /**
     * Feed-Forward Network
     * FFN(x) = GELU(xW1 + b1)W2 + b2
     */
    feedForward(x, seqLen, weights) {
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

        return output;
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
