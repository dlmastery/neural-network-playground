/**
 * Design Spaces Parameter Data
 * Comprehensive reference for architecture design parameters
 */

export const architectures = [
    {
        id: 'transformers',
        name: 'Transformers',
        icon: '⚡',
        color: '#6366f1',
        description: 'Attention-based architecture for sequence modeling',
        whenToUse: 'Sequential data, NLP, time series, any task needing global context',
        innovation: 'Self-attention mechanism that processes all positions in parallel',
        typicalSize: '10M - 175B parameters',
        yearIntroduced: 2017,
        parameters: [
            { name: 'Layers (L)', key: 'layers', type: 'slider', min: 1, max: 96, default: 12, step: 1, tip: '6-12 for most tasks; scale with data size', compute: 'high', impact: 'high' },
            { name: 'Model Dim (d_model)', key: 'd_model', type: 'slider', min: 64, max: 4096, default: 512, step: 64, tip: 'Power of 2; 512-1024 common', compute: 'high', impact: 'high' },
            { name: 'Attention Heads', key: 'heads', type: 'slider', min: 1, max: 64, default: 8, step: 1, tip: 'd_model/64 is typical', compute: 'medium', impact: 'medium' },
            { name: 'FFN Dim', key: 'ffn_dim', type: 'slider', min: 256, max: 16384, default: 2048, step: 256, tip: '4x d_model is standard', compute: 'high', impact: 'medium' },
            { name: 'Attention Type', key: 'attention_type', type: 'dropdown', options: ['MHA', 'MQA', 'GQA'], default: 'MHA', tip: 'GQA balances efficiency and quality', compute: 'varies', impact: 'medium' },
            { name: 'Positional Encoding', key: 'pos_encoding', type: 'dropdown', options: ['Sinusoidal', 'Learned', 'RoPE', 'ALiBi'], default: 'RoPE', tip: 'RoPE for length generalization', compute: 'low', impact: 'high' },
            { name: 'Dropout', key: 'dropout', type: 'slider', min: 0, max: 0.5, default: 0.1, step: 0.05, tip: 'Lower for larger models', compute: 'none', impact: 'medium' },
            { name: 'Layer Norm', key: 'layer_norm', type: 'dropdown', options: ['Pre-LN', 'Post-LN'], default: 'Pre-LN', tip: 'Pre-LN more stable', compute: 'low', impact: 'high' },
            { name: 'Activation', key: 'activation', type: 'dropdown', options: ['ReLU', 'GELU', 'SwiGLU'], default: 'SwiGLU', tip: 'SwiGLU for modern LLMs', compute: 'low', impact: 'medium' },
            { name: 'Head Dim', key: 'head_dim', type: 'computed', formula: 'd_model / heads', default: 64, tip: '64 is hardware optimal', compute: 'none', impact: 'low' }
        ]
    },
    {
        id: 'llms',
        name: 'Large Language Models',
        icon: '🗣️',
        color: '#8b5cf6',
        description: 'Scaled transformer models for text generation',
        whenToUse: 'Text generation, chat, code, reasoning tasks',
        innovation: 'Scaling laws - performance predictable from compute budget',
        typicalSize: '1B - 405B parameters',
        yearIntroduced: 2018,
        parameters: [
            { name: 'Total Params', key: 'total_params', type: 'dropdown', options: ['1B', '7B', '13B', '34B', '70B', '405B'], default: '7B', tip: 'Follow Chinchilla scaling', compute: 'varies', impact: 'high' },
            { name: 'Context Length', key: 'context_length', type: 'slider', min: 512, max: 128000, default: 4096, step: 512, tip: 'Longer = more memory', compute: 'high', impact: 'high' },
            { name: 'Vocab Size', key: 'vocab_size', type: 'slider', min: 32000, max: 256000, default: 50000, step: 1000, tip: 'Larger for multilingual', compute: 'medium', impact: 'medium' },
            { name: 'Tokenizer', key: 'tokenizer', type: 'dropdown', options: ['BPE', 'SentencePiece', 'Tiktoken'], default: 'BPE', tip: 'BPE with byte fallback is robust', compute: 'none', impact: 'medium' },
            { name: 'Architecture', key: 'architecture', type: 'dropdown', options: ['Dense', 'MoE'], default: 'Dense', tip: 'MoE for >70B params', compute: 'varies', impact: 'high' },
            { name: 'Expert Count (MoE)', key: 'num_experts', type: 'slider', min: 8, max: 128, default: 8, step: 8, tip: '8 experts is common', compute: 'medium', impact: 'high' },
            { name: 'Top-K Routing', key: 'top_k', type: 'slider', min: 1, max: 4, default: 2, step: 1, tip: '2 experts per token standard', compute: 'low', impact: 'medium' },
            { name: 'KV Cache', key: 'kv_cache', type: 'dropdown', options: ['Full', 'GQA', 'MQA', 'Paged'], default: 'GQA', tip: 'GQA reduces memory 4-8x', compute: 'varies', impact: 'medium' },
            { name: 'Tie Embeddings', key: 'tie_embeddings', type: 'toggle', default: true, tip: 'Saves ~15% params', compute: 'none', impact: 'low' },
            { name: 'RoPE Base', key: 'rope_base', type: 'slider', min: 10000, max: 1000000, default: 10000, step: 10000, tip: 'Increase for longer contexts', compute: 'none', impact: 'medium' }
        ]
    },
    {
        id: 'diffusion',
        name: 'Diffusion Models',
        icon: '🌊',
        color: '#ec4899',
        description: 'Denoising-based generative models',
        whenToUse: 'Image generation, audio synthesis, video generation',
        innovation: 'Learn to reverse a gradual noising process',
        typicalSize: '100M - 2B parameters',
        yearIntroduced: 2020,
        parameters: [
            { name: 'Timesteps (T)', key: 'timesteps', type: 'slider', min: 100, max: 1000, default: 1000, step: 100, tip: '1000 for training; fewer for inference', compute: 'high', impact: 'high' },
            { name: 'Noise Schedule', key: 'noise_schedule', type: 'dropdown', options: ['Linear', 'Cosine', 'Scaled Linear'], default: 'Cosine', tip: 'Cosine preserves more signal', compute: 'none', impact: 'high' },
            { name: 'Beta Start', key: 'beta_start', type: 'slider', min: 0.0001, max: 0.001, default: 0.0001, step: 0.0001, tip: 'Lower = less initial noise', compute: 'none', impact: 'medium' },
            { name: 'Beta End', key: 'beta_end', type: 'slider', min: 0.01, max: 0.1, default: 0.02, step: 0.01, tip: 'Higher = more final noise', compute: 'none', impact: 'medium' },
            { name: 'Sampler', key: 'sampler', type: 'dropdown', options: ['DDPM', 'DDIM', 'Euler', 'DPM++'], default: 'DDIM', tip: 'DDIM for fast inference', compute: 'varies', impact: 'high' },
            { name: 'Inference Steps', key: 'inference_steps', type: 'slider', min: 10, max: 100, default: 20, step: 5, tip: '20-50 for quality/speed balance', compute: 'high', impact: 'high' },
            { name: 'Guidance Scale', key: 'guidance_scale', type: 'slider', min: 1, max: 20, default: 7.5, step: 0.5, tip: '7-9 typical', compute: 'low', impact: 'high' },
            { name: 'U-Net Channels', key: 'unet_channels', type: 'slider', min: 64, max: 320, default: 128, step: 32, tip: 'Scale with image resolution', compute: 'high', impact: 'high' },
            { name: 'Attention Res', key: 'attention_res', type: 'dropdown', options: ['32, 16, 8', '16, 8', '8 only'], default: '16, 8', tip: 'Lower res = more global attention', compute: 'high', impact: 'medium' },
            { name: 'Conditioning', key: 'conditioning', type: 'dropdown', options: ['None', 'Class', 'Text', 'Image'], default: 'Text', tip: 'Text via CLIP embeddings', compute: 'medium', impact: 'high' }
        ]
    },
    {
        id: 'gnns',
        name: 'Graph Neural Networks',
        icon: '🕸️',
        color: '#10b981',
        description: 'Neural networks for graph-structured data',
        whenToUse: 'Molecules, social networks, knowledge graphs, recommendations',
        innovation: 'Message passing between nodes aggregates neighborhood info',
        typicalSize: '100K - 10M parameters',
        yearIntroduced: 2016,
        parameters: [
            { name: 'Layers', key: 'layers', type: 'slider', min: 2, max: 8, default: 3, step: 1, tip: '2-3 to avoid over-smoothing', compute: 'medium', impact: 'high' },
            { name: 'Hidden Dim', key: 'hidden_dim', type: 'slider', min: 32, max: 512, default: 128, step: 32, tip: 'Depends on graph complexity', compute: 'medium', impact: 'medium' },
            { name: 'Aggregation', key: 'aggregation', type: 'dropdown', options: ['Mean', 'Sum', 'Max', 'Attention'], default: 'Mean', tip: 'Sum for counting; Mean for averaging', compute: 'low', impact: 'high' },
            { name: 'Message Passing', key: 'message_passing', type: 'dropdown', options: ['GCN', 'GAT', 'GraphSAGE', 'GIN'], default: 'GCN', tip: 'GIN for expressivity', compute: 'varies', impact: 'high' },
            { name: 'Skip Connections', key: 'skip_connections', type: 'dropdown', options: ['None', 'Residual', 'Dense', 'JumpingKnowledge'], default: 'Residual', tip: 'JK for variable receptive fields', compute: 'low', impact: 'high' },
            { name: 'Normalization', key: 'normalization', type: 'dropdown', options: ['None', 'BatchNorm', 'LayerNorm', 'PairNorm'], default: 'BatchNorm', tip: 'PairNorm prevents over-smoothing', compute: 'low', impact: 'medium' },
            { name: 'Dropout', key: 'dropout', type: 'slider', min: 0, max: 0.5, default: 0.5, step: 0.1, tip: 'GNNs often need higher dropout', compute: 'none', impact: 'medium' },
            { name: 'Edge Dropout', key: 'edge_dropout', type: 'slider', min: 0, max: 0.5, default: 0.2, step: 0.1, tip: 'DropEdge helps generalization', compute: 'none', impact: 'medium' },
            { name: 'Readout', key: 'readout', type: 'dropdown', options: ['Mean', 'Sum', 'Set2Set', 'Virtual Node'], default: 'Mean', tip: 'Virtual node for global context', compute: 'low', impact: 'medium' },
            { name: 'Positional Encoding', key: 'pos_encoding', type: 'dropdown', options: ['None', 'Laplacian', 'Random Walk'], default: 'Laplacian', tip: 'For distinguishing symmetric structures', compute: 'medium', impact: 'high' }
        ]
    },
    {
        id: 'cnns',
        name: 'Convolutional Neural Networks',
        icon: '🔲',
        color: '#f59e0b',
        description: 'Spatial feature extractors for images',
        whenToUse: 'Image classification, object detection, segmentation',
        innovation: 'Local connectivity and weight sharing exploit spatial structure',
        typicalSize: '1M - 100M parameters',
        yearIntroduced: 1998,
        parameters: [
            { name: 'Kernel Size', key: 'kernel_size', type: 'dropdown', options: ['3x3', '5x5', '7x7'], default: '3x3', tip: '3x3 most efficient (VGG insight)', compute: 'varies', impact: 'medium' },
            { name: 'Depth', key: 'depth', type: 'slider', min: 5, max: 152, default: 50, step: 1, tip: 'ResNet-50 is versatile baseline', compute: 'high', impact: 'high' },
            { name: 'Width Multiplier', key: 'width_mult', type: 'slider', min: 0.25, max: 2.0, default: 1.0, step: 0.25, tip: 'Reduce for mobile', compute: 'high', impact: 'high' },
            { name: 'Pooling', key: 'pooling', type: 'dropdown', options: ['Max', 'Avg', 'Strided Conv'], default: 'Max', tip: 'Strided conv more flexible', compute: 'low', impact: 'medium' },
            { name: 'Skip Connections', key: 'skip_connections', type: 'dropdown', options: ['None', 'Residual', 'Dense'], default: 'Residual', tip: 'Essential for depth > 20', compute: 'low', impact: 'high' },
            { name: 'Normalization', key: 'normalization', type: 'dropdown', options: ['BatchNorm', 'LayerNorm', 'GroupNorm'], default: 'BatchNorm', tip: 'GroupNorm for small batches', compute: 'low', impact: 'medium' },
            { name: 'Activation', key: 'activation', type: 'dropdown', options: ['ReLU', 'LeakyReLU', 'SiLU', 'Mish'], default: 'ReLU', tip: 'SiLU for EfficientNet-style', compute: 'low', impact: 'low' },
            { name: 'Stem', key: 'stem', type: 'dropdown', options: ['7x7 stride 2', 'Patchify', 'Conv Stack'], default: '7x7 stride 2', tip: 'Patchify for ConvNeXt', compute: 'low', impact: 'medium' },
            { name: 'SE Blocks', key: 'se_blocks', type: 'toggle', default: true, tip: 'Squeeze-and-excite adds ~10% params', compute: 'low', impact: 'medium' },
            { name: 'Depthwise Separable', key: 'depthwise', type: 'toggle', default: false, tip: 'Essential for mobile efficiency', compute: 'low', impact: 'medium' }
        ]
    },
    {
        id: 'rnns',
        name: 'RNNs & LSTMs',
        icon: '↩️',
        color: '#ef4444',
        description: 'Recurrent networks for sequential data',
        whenToUse: 'Time series, speech, when order matters and context is local',
        innovation: 'Hidden state carries information across time steps',
        typicalSize: '100K - 50M parameters',
        yearIntroduced: 1997,
        parameters: [
            { name: 'Hidden Size', key: 'hidden_size', type: 'slider', min: 64, max: 2048, default: 512, step: 64, tip: 'Larger = more capacity', compute: 'high', impact: 'high' },
            { name: 'Layers', key: 'layers', type: 'slider', min: 1, max: 8, default: 2, step: 1, tip: '2-3 typical', compute: 'high', impact: 'medium' },
            { name: 'Cell Type', key: 'cell_type', type: 'dropdown', options: ['Vanilla', 'LSTM', 'GRU'], default: 'LSTM', tip: 'GRU simpler; LSTM more expressive', compute: 'varies', impact: 'high' },
            { name: 'Bidirectional', key: 'bidirectional', type: 'toggle', default: true, tip: 'Yes for classification; No for generation', compute: 'high', impact: 'high' },
            { name: 'Dropout', key: 'dropout', type: 'slider', min: 0, max: 0.5, default: 0.2, step: 0.1, tip: 'Applied between layers', compute: 'none', impact: 'medium' },
            { name: 'Input Dropout', key: 'input_dropout', type: 'slider', min: 0, max: 0.5, default: 0.1, step: 0.1, tip: 'On input embeddings', compute: 'none', impact: 'low' },
            { name: 'Residual', key: 'residual', type: 'toggle', default: true, tip: 'Helps deeper networks', compute: 'low', impact: 'medium' },
            { name: 'Layer Norm', key: 'layer_norm', type: 'toggle', default: true, tip: 'Stabilizes training', compute: 'low', impact: 'medium' },
            { name: 'Gradient Clipping', key: 'grad_clip', type: 'slider', min: 0.1, max: 10, default: 1.0, step: 0.1, tip: 'Essential for stability', compute: 'none', impact: 'high' },
            { name: 'Truncated BPTT', key: 'bptt_len', type: 'slider', min: 35, max: 200, default: 70, step: 5, tip: 'Longer = more context', compute: 'high', impact: 'medium' }
        ]
    },
    {
        id: 'ssms',
        name: 'SSMs / Mamba',
        icon: '〰️',
        color: '#06b6d4',
        description: 'State space models with selective mechanisms',
        whenToUse: 'Long sequences, when linear complexity is needed',
        innovation: 'Selective state spaces - input-dependent dynamics',
        typicalSize: '100M - 8B parameters',
        yearIntroduced: 2023,
        parameters: [
            { name: 'Model Dim (D)', key: 'd_model', type: 'slider', min: 256, max: 4096, default: 1024, step: 256, tip: 'Scale with task complexity', compute: 'high', impact: 'high' },
            { name: 'State Expansion (N)', key: 'state_expansion', type: 'slider', min: 16, max: 64, default: 16, step: 8, tip: 'Larger N = more expressivity', compute: 'high', impact: 'high' },
            { name: 'Conv Width (d_conv)', key: 'd_conv', type: 'slider', min: 2, max: 8, default: 4, step: 1, tip: '4 is standard from Mamba paper', compute: 'low', impact: 'low' },
            { name: 'Expansion Factor', key: 'expansion', type: 'slider', min: 1, max: 4, default: 2, step: 1, tip: 'Inner dim = expansion x model_dim', compute: 'high', impact: 'medium' },
            { name: 'Layers', key: 'layers', type: 'slider', min: 12, max: 64, default: 24, step: 4, tip: 'Similar scaling to Transformers', compute: 'high', impact: 'high' },
            { name: 'Discretization', key: 'discretization', type: 'dropdown', options: ['ZOH', 'Bilinear'], default: 'ZOH', tip: 'Zero-order hold is standard', compute: 'none', impact: 'low' },
            { name: 'Initialization', key: 'initialization', type: 'dropdown', options: ['S4D', 'Random'], default: 'S4D', tip: 'S4D crucial for stability', compute: 'none', impact: 'high' },
            { name: 'Activation', key: 'activation', type: 'dropdown', options: ['SiLU', 'GELU'], default: 'SiLU', tip: 'SiLU standard for Mamba', compute: 'low', impact: 'low' },
            { name: 'Normalization', key: 'normalization', type: 'dropdown', options: ['RMSNorm', 'LayerNorm'], default: 'RMSNorm', tip: 'RMSNorm before each block', compute: 'low', impact: 'low' },
            { name: 'Bidirectional', key: 'bidirectional', type: 'toggle', default: false, tip: 'Enable for non-causal tasks', compute: 'high', impact: 'medium' }
        ]
    },
    {
        id: 'vits',
        name: 'Vision Transformers',
        icon: '🖼️',
        color: '#a855f7',
        description: 'Transformers adapted for image understanding',
        whenToUse: 'Image classification, when you have lots of data',
        innovation: 'Treats image patches as tokens for attention',
        typicalSize: '10M - 600M parameters',
        yearIntroduced: 2020,
        parameters: [
            { name: 'Patch Size', key: 'patch_size', type: 'dropdown', options: ['4', '8', '14', '16', '32'], default: '16', tip: 'Smaller = more tokens, higher compute', compute: 'varies', impact: 'high' },
            { name: 'Image Size', key: 'image_size', type: 'slider', min: 224, max: 512, default: 224, step: 32, tip: '224 standard; 384/512 for fine-tuning', compute: 'high', impact: 'high' },
            { name: 'Embed Dim', key: 'embed_dim', type: 'slider', min: 192, max: 1024, default: 768, step: 64, tip: 'ViT-B: 768, ViT-L: 1024', compute: 'high', impact: 'high' },
            { name: 'Depth', key: 'depth', type: 'slider', min: 12, max: 32, default: 12, step: 1, tip: 'ViT-B: 12, ViT-L: 24', compute: 'high', impact: 'high' },
            { name: 'Heads', key: 'heads', type: 'slider', min: 3, max: 16, default: 12, step: 1, tip: 'embed_dim / 64 is common', compute: 'medium', impact: 'medium' },
            { name: 'MLP Ratio', key: 'mlp_ratio', type: 'slider', min: 2, max: 6, default: 4, step: 1, tip: 'FFN hidden = mlp_ratio x embed_dim', compute: 'high', impact: 'medium' },
            { name: 'Conv Stem', key: 'conv_stem', type: 'toggle', default: false, tip: 'Improves training stability', compute: 'low', impact: 'medium' },
            { name: 'Class Token', key: 'class_token', type: 'dropdown', options: ['CLS Token', 'GAP'], default: 'CLS Token', tip: 'GAP often simpler', compute: 'none', impact: 'low' },
            { name: 'Pos Encoding', key: 'pos_encoding', type: 'dropdown', options: ['Learned', 'Sinusoidal', '2D'], default: 'Learned', tip: '2D for variable resolutions', compute: 'none', impact: 'medium' },
            { name: 'Multi-stage', key: 'multi_stage', type: 'toggle', default: false, tip: 'Swin-style for dense prediction', compute: 'varies', impact: 'high' }
        ]
    },
    {
        id: 'vaes',
        name: 'Variational Autoencoders',
        icon: '🔄',
        color: '#22c55e',
        description: 'Probabilistic latent variable models',
        whenToUse: 'Representation learning, generation, anomaly detection',
        innovation: 'Learns compressed latent space with regularization',
        typicalSize: '100K - 10M parameters',
        yearIntroduced: 2013,
        parameters: [
            { name: 'Latent Dim', key: 'latent_dim', type: 'slider', min: 2, max: 512, default: 128, step: 8, tip: 'Higher = more capacity', compute: 'low', impact: 'high' },
            { name: 'Encoder Layers', key: 'encoder_layers', type: 'slider', min: 2, max: 6, default: 4, step: 1, tip: 'Symmetric with decoder', compute: 'medium', impact: 'medium' },
            { name: 'Decoder Layers', key: 'decoder_layers', type: 'slider', min: 2, max: 6, default: 4, step: 1, tip: 'Mirror encoder architecture', compute: 'medium', impact: 'medium' },
            { name: 'Hidden Dim', key: 'hidden_dim', type: 'slider', min: 64, max: 1024, default: 256, step: 64, tip: 'Gradually increase toward bottleneck', compute: 'medium', impact: 'medium' },
            { name: 'KL Weight (beta)', key: 'kl_weight', type: 'slider', min: 0.001, max: 10, default: 1.0, step: 0.1, tip: '<1 for reconstruction; >1 for disentangling', compute: 'none', impact: 'high' },
            { name: 'Annealing', key: 'annealing', type: 'toggle', default: true, tip: 'Gradually increase beta', compute: 'none', impact: 'medium' },
            { name: 'Prior', key: 'prior', type: 'dropdown', options: ['Standard Normal', 'Learned', 'VampPrior'], default: 'Standard Normal', tip: 'VampPrior for complex data', compute: 'varies', impact: 'medium' },
            { name: 'Decoder Output', key: 'decoder_output', type: 'dropdown', options: ['Gaussian', 'Bernoulli'], default: 'Gaussian', tip: 'Bernoulli for binary data', compute: 'none', impact: 'medium' },
            { name: 'Free Bits', key: 'free_bits', type: 'slider', min: 0, max: 2, default: 0, step: 0.1, tip: 'Prevents posterior collapse', compute: 'none', impact: 'medium' },
            { name: 'Architecture', key: 'architecture', type: 'dropdown', options: ['MLP', 'CNN', 'ResNet'], default: 'CNN', tip: 'CNN for images; MLP for tabular', compute: 'varies', impact: 'high' }
        ]
    },
    {
        id: 'gans',
        name: 'GANs',
        icon: '⚔️',
        color: '#f97316',
        description: 'Adversarial training for generation',
        whenToUse: 'Image generation when you need sharp outputs',
        innovation: 'Generator and discriminator compete to improve',
        typicalSize: '1M - 100M parameters',
        yearIntroduced: 2014,
        parameters: [
            { name: 'Latent Dim (z)', key: 'latent_dim', type: 'slider', min: 64, max: 512, default: 128, step: 32, tip: '128-256 typical', compute: 'low', impact: 'medium' },
            { name: 'Generator Layers', key: 'g_layers', type: 'slider', min: 4, max: 8, default: 5, step: 1, tip: 'Progressive growing helps', compute: 'high', impact: 'high' },
            { name: 'Discriminator Layers', key: 'd_layers', type: 'slider', min: 4, max: 8, default: 5, step: 1, tip: 'Match generator depth', compute: 'high', impact: 'high' },
            { name: 'Base Channels', key: 'base_channels', type: 'slider', min: 32, max: 128, default: 64, step: 16, tip: 'Double at each resolution', compute: 'high', impact: 'medium' },
            { name: 'Normalization (G)', key: 'g_norm', type: 'dropdown', options: ['BatchNorm', 'InstanceNorm', 'None'], default: 'BatchNorm', tip: 'InstanceNorm for StyleGAN', compute: 'low', impact: 'medium' },
            { name: 'Normalization (D)', key: 'd_norm', type: 'dropdown', options: ['None', 'SpectralNorm', 'LayerNorm'], default: 'SpectralNorm', tip: 'SpectralNorm stabilizes D', compute: 'low', impact: 'high' },
            { name: 'Loss', key: 'loss', type: 'dropdown', options: ['Vanilla', 'WGAN', 'WGAN-GP', 'Hinge'], default: 'Hinge', tip: 'Hinge loss stable and simple', compute: 'low', impact: 'high' },
            { name: 'Learning Rate (G)', key: 'g_lr', type: 'slider', min: 0.0001, max: 0.001, default: 0.0002, step: 0.0001, tip: 'Often lower than D', compute: 'none', impact: 'high' },
            { name: 'Learning Rate (D)', key: 'd_lr', type: 'slider', min: 0.0001, max: 0.001, default: 0.0002, step: 0.0001, tip: 'Match or higher than G', compute: 'none', impact: 'high' },
            { name: 'D Updates per G', key: 'd_updates', type: 'slider', min: 1, max: 5, default: 1, step: 1, tip: 'WGAN: 5; most others: 1', compute: 'varies', impact: 'medium' }
        ]
    }
];
