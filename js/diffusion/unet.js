/**
 * Tiny U-Net for MNIST Diffusion
 * ~100K parameters, designed for browser inference
 */

import { TensorOps } from './tensor-ops.js';

export class UNet {
    constructor() {
        this.weights = null;
        this.config = {
            imageSize: 28,
            inChannels: 1,
            outChannels: 1,
            baseChannels: 32,
            channelMult: [1, 2, 4], // 32, 64, 128
            numResBlocks: 1,
            timeEmbedDim: 64,
            numClasses: 10,
            classEmbedDim: 64
        };
    }

    /**
     * Load weights from binary file
     */
    async loadWeights(url) {
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();
        this.weights = this.parseWeights(buffer);
        console.log('[UNet] Weights loaded');
    }

    /**
     * Initialize random weights for training
     */
    initRandomWeights() {
        const cfg = this.config;
        this.weights = {};

        // Helper to create random weights
        const randn = (size, scale = 0.02) => {
            const arr = new Float32Array(size);
            for (let i = 0; i < size; i++) {
                // Box-Muller transform
                const u1 = Math.random();
                const u2 = Math.random();
                arr[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scale;
            }
            return arr;
        };

        const zeros = (size) => new Float32Array(size);
        const ones = (size) => {
            const arr = new Float32Array(size);
            arr.fill(1);
            return arr;
        };

        // Time embedding MLP
        this.weights.timeEmbed1 = randn(cfg.timeEmbedDim * cfg.timeEmbedDim * 4);
        this.weights.timeEmbed1Bias = zeros(cfg.timeEmbedDim * 4);
        this.weights.timeEmbed2 = randn(cfg.timeEmbedDim * 4 * cfg.timeEmbedDim);
        this.weights.timeEmbed2Bias = zeros(cfg.timeEmbedDim);

        // Class embedding
        this.weights.classEmbed = randn(cfg.numClasses * cfg.classEmbedDim);

        // Input conv
        this.weights.convIn = randn(3 * 3 * cfg.inChannels * cfg.baseChannels);
        this.weights.convInBias = zeros(cfg.baseChannels);

        // Encoder blocks (simplified - just conv + norm)
        const channels = cfg.channelMult.map(m => m * cfg.baseChannels);

        // Down block 1: 32 -> 64
        this.weights.down1Conv = randn(3 * 3 * channels[0] * channels[1]);
        this.weights.down1Bias = zeros(channels[1]);
        this.weights.down1Gamma = ones(channels[1]);
        this.weights.down1Beta = zeros(channels[1]);

        // Down block 2: 64 -> 128
        this.weights.down2Conv = randn(3 * 3 * channels[1] * channels[2]);
        this.weights.down2Bias = zeros(channels[2]);
        this.weights.down2Gamma = ones(channels[2]);
        this.weights.down2Beta = zeros(channels[2]);

        // Bottleneck
        this.weights.midConv = randn(3 * 3 * channels[2] * channels[2]);
        this.weights.midBias = zeros(channels[2]);
        this.weights.midGamma = ones(channels[2]);
        this.weights.midBeta = zeros(channels[2]);

        // Up block 1: 128+128 -> 64
        this.weights.up1Conv = randn(3 * 3 * (channels[2] * 2) * channels[1]);
        this.weights.up1Bias = zeros(channels[1]);
        this.weights.up1Gamma = ones(channels[1]);
        this.weights.up1Beta = zeros(channels[1]);

        // Up block 2: 64+64 -> 32
        this.weights.up2Conv = randn(3 * 3 * (channels[1] * 2) * channels[0]);
        this.weights.up2Bias = zeros(channels[0]);
        this.weights.up2Gamma = ones(channels[0]);
        this.weights.up2Beta = zeros(channels[0]);

        // Output conv
        this.weights.convOut = randn(3 * 3 * channels[0] * cfg.outChannels);
        this.weights.convOutBias = zeros(cfg.outChannels);

        console.log('[UNet] Random weights initialized');
    }

    /**
     * Parse weights from binary buffer
     */
    parseWeights(buffer) {
        // TODO: Implement actual weight parsing
        // For now, initialize random weights
        this.initRandomWeights();
        return this.weights;
    }

    /**
     * Forward pass
     * @param {Float32Array} x - Noisy image [28, 28, 1]
     * @param {Float32Array} timeEmb - Time embedding [64]
     * @param {number} classLabel - Class label (0-9) or -1 for unconditional
     * @returns {Float32Array} Predicted noise [28, 28, 1]
     */
    forward(x, timeEmb, classLabel = -1) {
        const cfg = this.config;
        const channels = cfg.channelMult.map(m => m * cfg.baseChannels);

        // Process time embedding through MLP
        let t = this.timeEmbedMLP(timeEmb);

        // Add class embedding if conditional
        if (classLabel >= 0 && classLabel < cfg.numClasses) {
            const classEmb = this.getClassEmbedding(classLabel);
            t = TensorOps.add(t, classEmb);
        }

        // Input conv: [28, 28, 1] -> [28, 28, 32]
        let h = this.conv2d(x, [28, 28, 1], this.weights.convIn, [3, 3, 1, channels[0]], this.weights.convInBias);
        h = TensorOps.silu(h);

        // Down 1: [28, 28, 32] -> [14, 14, 64]
        const skip1 = h;
        h = this.resBlock(h, [28, 28, channels[0]], t, 'down1');
        h = TensorOps.avgPool2x(h, [28, 28, channels[1]]);

        // Down 2: [14, 14, 64] -> [7, 7, 128]
        const skip2 = h;
        h = this.resBlock(h, [14, 14, channels[1]], t, 'down2');
        h = TensorOps.avgPool2x(h, [14, 14, channels[2]]);

        // Bottleneck: [7, 7, 128]
        h = this.resBlock(h, [7, 7, channels[2]], t, 'mid');

        // Up 1: [7, 7, 128] -> [14, 14, 64]
        h = TensorOps.upsample2x(h, [7, 7, channels[2]]);
        h = TensorOps.concat(h, skip2, [14, 14, channels[2]]);
        h = this.resBlock(h, [14, 14, channels[2] * 2], t, 'up1');

        // Up 2: [14, 14, 64] -> [28, 28, 32]
        h = TensorOps.upsample2x(h, [14, 14, channels[1]]);
        h = TensorOps.concat(h, skip1, [28, 28, channels[1]]);
        h = this.resBlock(h, [28, 28, channels[1] * 2], t, 'up2');

        // Output conv: [28, 28, 32] -> [28, 28, 1]
        h = this.conv2d(h, [28, 28, channels[0]], this.weights.convOut, [3, 3, channels[0], 1], this.weights.convOutBias);

        return h;
    }

    /**
     * Time embedding MLP
     */
    timeEmbedMLP(timeEmb) {
        const dim = this.config.timeEmbedDim;

        // Linear 1: [64] -> [256]
        let h = this.linear(timeEmb, this.weights.timeEmbed1, this.weights.timeEmbed1Bias, dim, dim * 4);
        h = TensorOps.silu(h);

        // Linear 2: [256] -> [64]
        h = this.linear(h, this.weights.timeEmbed2, this.weights.timeEmbed2Bias, dim * 4, dim);

        return h;
    }

    /**
     * Get class embedding
     */
    getClassEmbedding(classLabel) {
        const dim = this.config.classEmbedDim;
        const start = classLabel * dim;
        return this.weights.classEmbed.slice(start, start + dim);
    }

    /**
     * Residual block with time conditioning
     */
    resBlock(x, shape, timeEmb, prefix) {
        const [H, W, C] = shape;
        const outC = this.weights[`${prefix}Conv`].length / (9 * C);

        // GroupNorm + SiLU + Conv
        let h = TensorOps.groupNorm(x, shape, 8, this.weights[`${prefix}Gamma`], this.weights[`${prefix}Beta`]);
        h = TensorOps.silu(h);
        h = this.conv2d(h, shape, this.weights[`${prefix}Conv`], [3, 3, C, outC], this.weights[`${prefix}Bias`]);

        // Add time embedding (broadcast across spatial dims)
        // Simplified: just scale by time embedding magnitude
        const timeScale = timeEmb.reduce((a, b) => a + b, 0) / timeEmb.length;
        for (let i = 0; i < h.length; i++) {
            h[i] *= (1 + timeScale * 0.1);
        }

        return h;
    }

    /**
     * Conv2D wrapper
     */
    conv2d(input, inputShape, kernel, kernelShape, bias) {
        let output = TensorOps.conv2d(input, kernel, inputShape, kernelShape, 1, 'same');

        // Add bias
        const [H, W, C] = [inputShape[0], inputShape[1], kernelShape[3]];
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                for (let c = 0; c < C; c++) {
                    output[(h * W + w) * C + c] += bias[c];
                }
            }
        }

        return output;
    }

    /**
     * Linear layer
     */
    linear(input, weights, bias, inDim, outDim) {
        const output = new Float32Array(outDim);

        for (let o = 0; o < outDim; o++) {
            let sum = bias[o];
            for (let i = 0; i < inDim; i++) {
                sum += input[i] * weights[i * outDim + o];
            }
            output[o] = sum;
        }

        return output;
    }

    /**
     * Count total parameters
     */
    countParams() {
        if (!this.weights) return 0;

        let total = 0;
        for (const key in this.weights) {
            total += this.weights[key].length;
        }
        return total;
    }
}

export const unet = new UNet();
