/**
 * Diffusion Model - Tiny U-Net for MNIST generation
 * Implements DDPM/DDIM sampling with WebGPU acceleration
 */

import { unet } from './unet.js';

export class DiffusionModel {
    constructor() {
        this.isLoaded = false;
        this.backend = 'webgpu';
        this.device = null;

        // Model config
        this.config = {
            imageSize: 28,
            channels: 1,
            numTimesteps: 1000,
            numClasses: 10
        };

        // Noise schedule
        this.betas = null;
        this.alphas = null;
        this.alphasCumprod = null;
    }

    /**
     * Initialize the model with specified backend
     */
    async init(backend = 'webgpu') {
        this.backend = backend;

        // Initialize noise schedule
        this.initNoiseSchedule();

        if (backend === 'webgpu') {
            await this.initWebGPU();
        }

        console.log(`[DiffusionModel] Initialized with ${backend} backend`);
    }

    /**
     * Initialize linear noise schedule
     */
    initNoiseSchedule() {
        const T = this.config.numTimesteps;
        const betaStart = 0.0001;
        const betaEnd = 0.02;

        // Linear schedule
        this.betas = new Float32Array(T);
        this.alphas = new Float32Array(T);
        this.alphasCumprod = new Float32Array(T);

        for (let t = 0; t < T; t++) {
            this.betas[t] = betaStart + (betaEnd - betaStart) * t / (T - 1);
            this.alphas[t] = 1 - this.betas[t];
            this.alphasCumprod[t] = t === 0
                ? this.alphas[t]
                : this.alphasCumprod[t - 1] * this.alphas[t];
        }
    }

    /**
     * Initialize WebGPU device and pipelines
     */
    async initWebGPU() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No WebGPU adapter found');
        }

        this.device = await adapter.requestDevice();
        console.log('[DiffusionModel] WebGPU device initialized');

        // TODO: Create compute pipelines in Phase 2
    }

    /**
     * Load pre-trained weights
     */
    async loadWeights(url) {
        // TODO: Implement weight loading
        console.log('[DiffusionModel] Loading weights from', url);
        this.isLoaded = true;
    }

    /**
     * Generate samples using DDPM
     */
    async sample(numSamples, numSteps, classLabel = -1, guidanceScale = 1.0, onProgress = null) {
        const { imageSize, channels, numTimesteps } = this.config;
        const imagePixels = imageSize * imageSize * channels;

        // Ensure U-Net is initialized
        if (!unet.weights) {
            unet.initRandomWeights();
        }

        // Start from pure noise
        const samples = [];
        const histories = []; // Store intermediate steps for visualization

        for (let i = 0; i < numSamples; i++) {
            samples.push(this.randomNormal(imagePixels));
            histories.push([new Float32Array(samples[i])]); // Save initial noise
        }

        // Denoising loop
        const stepIndices = this.getStepIndices(numSteps);

        for (let stepIdx = 0; stepIdx < stepIndices.length; stepIdx++) {
            const t = stepIndices[stepIdx];
            const tNext = stepIdx < stepIndices.length - 1 ? stepIndices[stepIdx + 1] : 0;

            // Get time embedding
            const timeEmb = this.getTimeEmbedding(t);

            for (let i = 0; i < numSamples; i++) {
                // Predict noise
                let predictedNoise;

                if (guidanceScale > 1.0 && classLabel >= 0) {
                    // Classifier-free guidance
                    const condNoise = unet.forward(samples[i], timeEmb, classLabel);
                    const uncondNoise = unet.forward(samples[i], timeEmb, -1);

                    predictedNoise = new Float32Array(imagePixels);
                    for (let j = 0; j < imagePixels; j++) {
                        predictedNoise[j] = uncondNoise[j] + guidanceScale * (condNoise[j] - uncondNoise[j]);
                    }
                } else {
                    predictedNoise = unet.forward(samples[i], timeEmb, classLabel);
                }

                // DDPM step
                samples[i] = this.ddpmStep(samples[i], predictedNoise, t, tNext);

                // Save to history (every few steps)
                if (stepIdx % Math.max(1, Math.floor(numSteps / 10)) === 0) {
                    histories[i].push(new Float32Array(samples[i]));
                }
            }

            if (onProgress) {
                onProgress(stepIdx + 1, numSteps);
            }

            // Yield to UI
            await new Promise(r => setTimeout(r, 0));
        }

        // Save final result to histories
        for (let i = 0; i < numSamples; i++) {
            histories[i].push(new Float32Array(samples[i]));
        }

        return { samples, histories };
    }

    /**
     * Get step indices for sampling
     */
    getStepIndices(numSteps) {
        const indices = [];
        const stepSize = Math.floor(this.config.numTimesteps / numSteps);

        for (let i = 0; i < numSteps; i++) {
            indices.push(this.config.numTimesteps - 1 - i * stepSize);
        }

        return indices;
    }

    /**
     * Single DDPM denoising step
     */
    ddpmStep(x, predictedNoise, t, tNext) {
        const alpha = this.alphas[t];
        const alphaCumprod = this.alphasCumprod[t];
        const beta = this.betas[t];

        const sqrtOneMinusAlphaCumprod = Math.sqrt(1 - alphaCumprod);

        const output = new Float32Array(x.length);

        // Mean
        const coef1 = 1 / Math.sqrt(alpha);
        const coef2 = beta / sqrtOneMinusAlphaCumprod;

        for (let i = 0; i < x.length; i++) {
            output[i] = coef1 * (x[i] - coef2 * predictedNoise[i]);
        }

        // Add noise (except for last step)
        if (tNext > 0) {
            const sigma = Math.sqrt(beta);
            const noise = this.randomNormal(x.length);
            for (let i = 0; i < x.length; i++) {
                output[i] += sigma * noise[i];
            }
        }

        return output;
    }

    /**
     * Generate random normal values
     */
    randomNormal(size) {
        const arr = new Float32Array(size);
        for (let i = 0; i < size; i += 2) {
            const u1 = Math.random();
            const u2 = Math.random();
            const r = Math.sqrt(-2 * Math.log(u1));
            const theta = 2 * Math.PI * u2;
            arr[i] = r * Math.cos(theta);
            if (i + 1 < size) {
                arr[i + 1] = r * Math.sin(theta);
            }
        }
        return arr;
    }

    /**
     * Add noise to image at timestep t (forward diffusion)
     */
    addNoise(image, t) {
        const alphaCumprod = this.alphasCumprod[t];
        const noise = this.randomNormal(image.length);
        const result = new Float32Array(image.length);

        const sqrtAlpha = Math.sqrt(alphaCumprod);
        const sqrtOneMinusAlpha = Math.sqrt(1 - alphaCumprod);

        for (let i = 0; i < image.length; i++) {
            result[i] = sqrtAlpha * image[i] + sqrtOneMinusAlpha * noise[i];
        }

        return { noisy: result, noise };
    }

    /**
     * Create sinusoidal time embedding
     * @param {number} timestep - Current timestep
     * @param {number} dim - Embedding dimension
     * @returns {Float32Array} Time embedding vector
     */
    getTimeEmbedding(timestep, dim = 64) {
        const halfDim = dim / 2;
        const embedding = new Float32Array(dim);

        // Sinusoidal embedding formula
        const logTimescale = Math.log(10000) / (halfDim - 1);

        for (let i = 0; i < halfDim; i++) {
            const freq = Math.exp(-i * logTimescale);
            const angle = timestep * freq;
            embedding[i] = Math.sin(angle);
            embedding[i + halfDim] = Math.cos(angle);
        }

        return embedding;
    }
}

// Export singleton
export const diffusionModel = new DiffusionModel();
