/**
 * Transformer Model Wrapper
 * Wraps Transformers.js for text generation with WebGPU support
 */

// Transformers.js is loaded dynamically
let transformersModule = null;

export class TransformerModel {
    constructor() {
        this.generator = null;
        this.tokenizer = null;
        this.model = null;
        this.modelId = 'Xenova/distilgpt2';
        this.device = 'wasm'; // 'webgpu' or 'wasm'
        this.isLoading = false;
        this.isLoaded = false;
        this.onProgress = null;
        this.onStatusChange = null;

        // Model info for display
        this.modelInfo = {
            'Xenova/distilgpt2': {
                name: 'DistilGPT2',
                params: '82M',
                layers: 6,
                hidden: 768,
                heads: 12,
                vocab: 50257
            },
            'Xenova/gpt2': {
                name: 'GPT-2',
                params: '124M',
                layers: 12,
                hidden: 768,
                heads: 12,
                vocab: 50257
            }
        };
    }

    /**
     * Check if WebGPU is available
     */
    async checkWebGPU() {
        if (!navigator.gpu) {
            return false;
        }
        try {
            const adapter = await navigator.gpu.requestAdapter();
            return adapter !== null;
        } catch (e) {
            console.warn('WebGPU check failed:', e);
            return false;
        }
    }

    /**
     * Load Transformers.js dynamically
     */
    async loadTransformersJS() {
        if (transformersModule) {
            return transformersModule;
        }

        this.updateStatus('Loading Transformers.js...');

        try {
            // Dynamic import from CDN
            transformersModule = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.0');

            // Configure environment
            const { env } = transformersModule;
            env.allowLocalModels = false;
            env.useBrowserCache = true;

            return transformersModule;
        } catch (error) {
            console.error('Failed to load Transformers.js:', error);
            throw error;
        }
    }

    /**
     * Initialize the model
     * @param {string} modelId - Model identifier
     * @param {string} device - 'webgpu' or 'wasm'
     */
    async init(modelId = 'Xenova/distilgpt2', device = 'wasm') {
        if (this.isLoading) {
            console.warn('Model is already loading');
            return;
        }

        this.isLoading = true;
        this.isLoaded = false;
        this.modelId = modelId;

        try {
            // Load Transformers.js
            const { pipeline, AutoTokenizer } = await this.loadTransformersJS();

            // Check WebGPU availability
            if (device === 'webgpu') {
                const hasWebGPU = await this.checkWebGPU();
                if (!hasWebGPU) {
                    console.warn('WebGPU not available, falling back to WASM');
                    device = 'wasm';
                }
            }
            this.device = device;

            this.updateStatus(`Loading ${this.getModelInfo().name}...`);

            // Load tokenizer first
            this.tokenizer = await AutoTokenizer.from_pretrained(modelId, {
                progress_callback: (progress) => this.handleProgress(progress)
            });

            // Create text generation pipeline
            this.generator = await pipeline('text-generation', modelId, {
                device: device,
                progress_callback: (progress) => this.handleProgress(progress)
            });

            this.isLoaded = true;
            this.updateStatus('Ready');
            console.log(`Model ${modelId} loaded on ${device}`);

        } catch (error) {
            console.error('Failed to initialize model:', error);
            this.updateStatus('Error loading model');
            throw error;
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Handle progress updates during loading
     */
    handleProgress(progress) {
        if (this.onProgress) {
            // Normalize progress object
            let percentage = 0;
            let status = '';

            if (progress.status === 'initiate') {
                status = `Downloading ${progress.file || 'model'}...`;
                percentage = 0;
            } else if (progress.status === 'download') {
                percentage = progress.progress || 0;
                status = `Downloading... ${Math.round(percentage)}%`;
            } else if (progress.status === 'progress') {
                percentage = progress.progress || 0;
                status = `Loading... ${Math.round(percentage)}%`;
            } else if (progress.status === 'done') {
                percentage = 100;
                status = 'Processing...';
            } else if (progress.status === 'ready') {
                percentage = 100;
                status = 'Ready';
            }

            this.onProgress({ percentage, status, raw: progress });
        }
    }

    /**
     * Update status display
     */
    updateStatus(status) {
        if (this.onStatusChange) {
            this.onStatusChange(status);
        }
    }

    /**
     * Get model info for current model
     */
    getModelInfo() {
        return this.modelInfo[this.modelId] || this.modelInfo['Xenova/distilgpt2'];
    }

    /**
     * Tokenize text
     * @param {string} text - Text to tokenize
     * @returns {object} - { tokens: string[], ids: number[] }
     */
    tokenize(text) {
        if (!this.tokenizer) {
            return { tokens: [], ids: [] };
        }

        const encoded = this.tokenizer.encode(text);
        const tokens = [];

        // Decode each token individually to show the token strings
        for (let i = 0; i < encoded.length; i++) {
            const tokenStr = this.tokenizer.decode([encoded[i]]);
            tokens.push(tokenStr);
        }

        return {
            tokens,
            ids: Array.from(encoded)
        };
    }

    /**
     * Generate text
     * @param {string} prompt - Input text
     * @param {object} options - Generation options
     * @returns {AsyncGenerator} - Yields generated tokens
     */
    async *generate(prompt, options = {}) {
        if (!this.generator) {
            throw new Error('Model not loaded');
        }

        const {
            maxNewTokens = 50,
            temperature = 0.7,
            topK = 50,
            topP = 0.95,
            doSample = true,
            onToken = null
        } = options;

        this.updateStatus('Generating...');

        try {
            // Use streaming generation
            const streamer = new TextStreamer(this.tokenizer, {
                skip_prompt: true,
                skip_special_tokens: true
            });

            let generatedText = prompt;
            let tokenCount = 0;

            // Generate with streaming callback
            const output = await this.generator(prompt, {
                max_new_tokens: maxNewTokens,
                temperature,
                top_k: topK,
                top_p: topP,
                do_sample: doSample,
                return_full_text: true,
                callback_function: (x) => {
                    // This gets called for each token
                    const newToken = this.tokenizer.decode([x[0].output_token_ids.at(-1)]);
                    tokenCount++;
                    if (onToken) {
                        onToken(newToken, tokenCount);
                    }
                }
            });

            // Yield the final output
            if (output && output[0]) {
                yield {
                    text: output[0].generated_text,
                    done: true,
                    tokenCount
                };
            }

            this.updateStatus('Ready');
        } catch (error) {
            console.error('Generation error:', error);
            this.updateStatus('Error');
            throw error;
        }
    }

    /**
     * Generate text synchronously (returns full result)
     * @param {string} prompt - Input text
     * @param {object} options - Generation options
     * @returns {Promise<string>} - Generated text
     */
    async generateSync(prompt, options = {}) {
        if (!this.generator) {
            throw new Error('Model not loaded');
        }

        const {
            maxNewTokens = 50,
            temperature = 0.7,
            topK = 50,
            topP = 0.95,
            doSample = true
        } = options;

        this.updateStatus('Generating...');

        try {
            const output = await this.generator(prompt, {
                max_new_tokens: maxNewTokens,
                temperature,
                top_k: topK,
                top_p: topP,
                do_sample: doSample,
                return_full_text: true
            });

            this.updateStatus('Ready');

            if (output && output[0]) {
                return output[0].generated_text;
            }
            return prompt;
        } catch (error) {
            console.error('Generation error:', error);
            this.updateStatus('Error');
            throw error;
        }
    }

    /**
     * Dispose of resources
     */
    dispose() {
        this.generator = null;
        this.tokenizer = null;
        this.model = null;
        this.isLoaded = false;
    }
}

/**
 * Simple text streamer for token-by-token output
 */
class TextStreamer {
    constructor(tokenizer, options = {}) {
        this.tokenizer = tokenizer;
        this.skipPrompt = options.skip_prompt || false;
        this.skipSpecialTokens = options.skip_special_tokens || true;
        this.tokens = [];
    }

    put(tokenIds) {
        this.tokens.push(...tokenIds);
    }

    end() {
        return this.tokenizer.decode(this.tokens, {
            skip_special_tokens: this.skipSpecialTokens
        });
    }
}

// Export singleton instance
export const transformerModel = new TransformerModel();
