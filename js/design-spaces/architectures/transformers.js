/**
 * Transformers Architecture Configuration
 * Attention-based architecture for sequence modeling
 */

export const transformers = {
    id: 'transformers',
    name: 'Transformers',
    icon: '⚡',
    color: '#6366f1',
    description: 'Attention-based architecture for sequence modeling',
    fullDescription: 'Transformers revolutionized sequence modeling by replacing recurrence with self-attention, enabling parallel processing and capturing long-range dependencies. The architecture consists of stacked encoder/decoder blocks with multi-head attention and feedforward networks.',
    whenToUse: 'Sequential data, NLP, time series, any task needing global context',
    innovation: 'Self-attention mechanism that processes all positions in parallel',
    typicalSize: '10M - 175B parameters',
    yearIntroduced: 2017,
    keyPapers: [
        { title: 'Attention Is All You Need', authors: 'Vaswani et al.', year: 2017, arxiv: '1706.03762' },
        { title: 'BERT: Pre-training of Deep Bidirectional Transformers', authors: 'Devlin et al.', year: 2018, arxiv: '1810.04805' },
        { title: 'Language Models are Few-Shot Learners (GPT-3)', authors: 'Brown et al.', year: 2020, arxiv: '2005.14165' }
    ],
    scalingGuidelines: {
        small: { params: '<100M', layers: '6-8', d_model: '256-512', heads: '4-8', note: 'Good for single-task fine-tuning' },
        medium: { params: '100M-1B', layers: '12-24', d_model: '768-1024', heads: '12-16', note: 'Strong general-purpose models' },
        large: { params: '>1B', layers: '24-96', d_model: '2048-4096', heads: '16-64', note: 'Emergent capabilities appear' }
    },
    parameters: [
        {
            name: 'Layers (L)',
            key: 'layers',
            category: 'architecture',
            type: 'slider',
            min: 1,
            max: 96,
            default: 12,
            step: 1,
            description: 'Number of transformer blocks stacked sequentially. Each layer adds representational capacity and increases the effective receptive field for capturing patterns.',
            tip: '6-12 for most tasks; scale with data size',
            compute: 'high',
            impact: 'high',
            tradeoffs: {
                increase: ['Deeper representations', 'Better long-range modeling', 'More emergent capabilities'],
                decrease: ['Faster training/inference', 'Less memory', 'Easier optimization']
            },
            research: 'Scaling laws (Kaplan et al., 2020) show depth scales sublinearly with compute budget. Deep networks benefit from Pre-LN for stability.',
            recommendations: {
                small: '6 layers for simple tasks',
                medium: '12 layers (BERT-base standard)',
                large: '24-48 layers for complex reasoning'
            }
        },
        {
            name: 'Model Dim (d_model)',
            key: 'd_model',
            category: 'architecture',
            type: 'slider',
            min: 64,
            max: 4096,
            default: 512,
            step: 64,
            description: 'Hidden dimension throughout the model. This is the "width" of your network - larger values can encode richer representations but cost quadratically in attention.',
            tip: 'Power of 2; 512-1024 common',
            compute: 'high',
            impact: 'high',
            tradeoffs: {
                increase: ['Richer representations', 'More nuanced features', 'Better for complex data'],
                decrease: ['Much faster (O(d²) attention)', 'Less memory', 'Better for simple tasks']
            },
            research: 'GPT-3 uses d_model=12288. Chinchilla found width should scale roughly √N for N parameters.',
            recommendations: {
                small: '256-512 for classification',
                medium: '768-1024 (BERT/GPT-2 range)',
                large: '2048-4096+ for LLMs'
            }
        },
        {
            name: 'Attention Heads',
            key: 'heads',
            category: 'attention',
            type: 'slider',
            min: 1,
            max: 64,
            default: 8,
            step: 1,
            description: 'Number of parallel attention patterns. Each head can learn different relationship types (syntactic, semantic, positional). Head dimension = d_model / heads.',
            tip: 'd_model/64 is typical; ensures head_dim=64',
            compute: 'medium',
            impact: 'medium',
            tradeoffs: {
                increase: ['More diverse attention patterns', 'Better multi-aspect modeling'],
                decrease: ['Larger head dimension', 'Potentially stronger per-head attention']
            },
            research: 'Michel et al. (2019) found many heads can be pruned post-training. Keep head_dim=64 for hardware efficiency (tensor core alignment).',
            recommendations: {
                small: '4-8 heads',
                medium: '12 heads (BERT-base)',
                large: '32-64 heads for very wide models'
            }
        },
        {
            name: 'FFN Dim',
            key: 'ffn_dim',
            category: 'architecture',
            type: 'slider',
            min: 256,
            max: 16384,
            default: 2048,
            step: 256,
            description: 'Hidden dimension of the feedforward network in each transformer block. This is where most parameters live and where "knowledge" is stored.',
            tip: '4x d_model is standard; some use 8/3x for SwiGLU',
            compute: 'high',
            impact: 'medium',
            tradeoffs: {
                increase: ['More stored knowledge', 'Better memorization', 'Richer transformations'],
                decrease: ['Faster inference', 'Smaller model size', 'Less overfitting risk']
            },
            research: 'PaLM uses 4x, LLaMA uses ~2.7x with SwiGLU (since SwiGLU has 3 weight matrices vs 2).',
            recommendations: {
                small: '4x d_model',
                medium: '4x d_model',
                large: '8/3x d_model with SwiGLU activation'
            }
        },
        {
            name: 'Attention Type',
            key: 'attention_type',
            category: 'attention',
            type: 'dropdown',
            options: ['MHA', 'MQA', 'GQA'],
            default: 'MHA',
            description: 'Multi-Head (MHA), Multi-Query (MQA), or Grouped-Query (GQA) attention. MQA/GQA share key-value heads to reduce KV-cache memory during inference.',
            tip: 'GQA balances efficiency and quality',
            compute: 'varies',
            impact: 'medium',
            tradeoffs: {
                MHA: ['Best quality', 'Full expressiveness', 'Highest memory use'],
                MQA: ['Fastest inference', 'Minimal KV-cache', 'Slight quality drop'],
                GQA: ['Good quality/speed balance', 'Moderate KV-cache', 'Recommended for production']
            },
            research: 'GQA (Ainslie et al., 2023) achieves MHA quality with MQA speed. LLaMA-2 70B uses GQA with 8 KV heads.',
            recommendations: {
                small: 'MHA (memory not a bottleneck)',
                medium: 'GQA with 4-8 KV groups',
                large: 'GQA essential for serving'
            }
        },
        {
            name: 'Positional Encoding',
            key: 'pos_encoding',
            category: 'architecture',
            type: 'dropdown',
            options: ['Sinusoidal', 'Learned', 'RoPE', 'ALiBi'],
            default: 'RoPE',
            description: 'How position information is injected. Critical for sequence understanding since attention is permutation-invariant without position signals.',
            tip: 'RoPE for length generalization; ALiBi for extrapolation',
            compute: 'low',
            impact: 'high',
            tradeoffs: {
                Sinusoidal: ['No learned params', 'Fixed pattern', 'Poor extrapolation'],
                Learned: ['Flexible', 'Fixed max length', 'Needs fine-tuning to extend'],
                RoPE: ['Relative positions', 'Good extrapolation', 'Most popular for LLMs'],
                ALiBi: ['Linear bias', 'Best extrapolation', 'No position embeddings needed']
            },
            research: 'RoPE (Su et al., 2021) enables length extrapolation via rotation. Used by LLaMA, Mistral. ALiBi (Press et al., 2021) adds attention bias.',
            recommendations: {
                small: 'Learned or Sinusoidal',
                medium: 'RoPE',
                large: 'RoPE with base scaling for long context'
            }
        },
        {
            name: 'Dropout',
            key: 'dropout',
            category: 'regularization',
            type: 'slider',
            min: 0,
            max: 0.5,
            default: 0.1,
            step: 0.05,
            description: 'Randomly zeros activations during training to prevent co-adaptation. Applied after attention and FFN layers.',
            tip: 'Lower for larger models (they regularize via scale)',
            compute: 'none',
            impact: 'medium',
            tradeoffs: {
                increase: ['Stronger regularization', 'Better generalization on small data'],
                decrease: ['Faster convergence', 'Better for large models/datasets']
            },
            research: 'Large models (>1B params) often use dropout=0 as the data volume provides implicit regularization.',
            recommendations: {
                small: '0.1-0.3',
                medium: '0.1',
                large: '0.0-0.1'
            }
        },
        {
            name: 'Layer Norm',
            key: 'layer_norm',
            category: 'training',
            type: 'dropdown',
            options: ['Pre-LN', 'Post-LN', 'RMSNorm'],
            default: 'Pre-LN',
            description: 'Normalization placement and type. Pre-LN applies norm before attention/FFN (more stable). RMSNorm is faster, omitting mean centering.',
            tip: 'Pre-LN more stable; RMSNorm for efficiency',
            compute: 'low',
            impact: 'high',
            tradeoffs: {
                'Pre-LN': ['Stable training', 'Easier optimization', 'Standard choice'],
                'Post-LN': ['Original design', 'Needs careful LR tuning', 'Slightly better final quality'],
                'RMSNorm': ['15% faster', 'Similar quality', 'Used by LLaMA/Mistral']
            },
            research: 'Xiong et al. (2020) showed Pre-LN enables training without warmup. RMSNorm (Zhang & Sennrich, 2019) removes mean computation.',
            recommendations: {
                small: 'Pre-LN or Post-LN',
                medium: 'Pre-LN',
                large: 'RMSNorm (efficiency matters at scale)'
            }
        },
        {
            name: 'Activation',
            key: 'activation',
            category: 'architecture',
            type: 'dropdown',
            options: ['ReLU', 'GELU', 'SwiGLU', 'GeGLU'],
            default: 'SwiGLU',
            description: 'Non-linearity in FFN layers. Gated variants (SwiGLU, GeGLU) add a gating mechanism that improves performance at ~50% more FFN parameters.',
            tip: 'SwiGLU for modern LLMs; GELU for BERT-style',
            compute: 'low',
            impact: 'medium',
            tradeoffs: {
                ReLU: ['Simple', 'Fast', 'Can have dead neurons'],
                GELU: ['Smooth', 'BERT standard', 'Good general choice'],
                SwiGLU: ['Best quality', 'Gated mechanism', 'More parameters'],
                GeGLU: ['Similar to SwiGLU', 'GELU-based gate', 'Alternative option']
            },
            research: 'Shazeer (2020) showed GLU variants outperform ReLU/GELU. PaLM, LLaMA, Mistral all use SwiGLU.',
            recommendations: {
                small: 'GELU',
                medium: 'GELU or SwiGLU',
                large: 'SwiGLU'
            }
        },
        {
            name: 'Head Dim',
            key: 'head_dim',
            category: 'efficiency',
            type: 'computed',
            formula: 'd_model / heads',
            default: 64,
            description: 'Dimension per attention head. This value should typically be 64 or 128 for optimal hardware utilization on modern GPUs.',
            tip: '64 is hardware optimal; 128 for quality',
            compute: 'none',
            impact: 'low',
            tradeoffs: {
                increase: ['Richer per-head representations', 'May improve quality'],
                decrease: ['More heads possible', 'Better hardware alignment at 64']
            },
            research: 'NVIDIA tensor cores are optimized for dimensions divisible by 8, with 64 being particularly efficient.',
            recommendations: {
                small: '64',
                medium: '64',
                large: '64 or 128'
            }
        },
        {
            name: 'Flash Attention',
            key: 'flash_attention',
            category: 'efficiency',
            type: 'toggle',
            default: true,
            description: 'Memory-efficient attention algorithm that avoids materializing the full attention matrix. Reduces memory from O(n²) to O(n) and is 2-4x faster.',
            tip: 'Essential for long sequences; standard in production',
            compute: 'low',
            impact: 'high',
            tradeoffs: {
                enabled: ['2-4x faster', 'O(n) memory', 'Longer sequences possible', 'Standard practice'],
                disabled: ['Full attention matrix', 'Better for debugging', 'Some custom attention patterns']
            },
            research: 'Flash Attention (Dao et al., 2022) uses tiling to compute attention in SRAM. Flash Attention 2 adds further optimizations.',
            recommendations: {
                default: 'Enable for all production use',
                debugging: 'Disable to inspect attention patterns',
                custom_attention: 'May need custom implementation'
            }
        },
        {
            name: 'Weight Decay',
            key: 'weight_decay',
            category: 'training',
            type: 'slider',
            min: 0,
            max: 0.3,
            default: 0.1,
            step: 0.01,
            description: 'L2 regularization coefficient in AdamW optimizer. Applied to all weights except biases and layer norm parameters.',
            tip: '0.1 standard for LLMs; helps generalization',
            compute: 'none',
            impact: 'medium',
            tradeoffs: {
                increase: ['Stronger regularization', 'Prevents large weights', 'May slow convergence'],
                decrease: ['Faster convergence', 'Risk of overfitting', 'Larger weight magnitudes']
            },
            research: 'Loshchilov & Hutter (2017) showed decoupled weight decay (AdamW) works better than L2 in Adam. LLaMA uses 0.1.',
            recommendations: {
                small_data: '0.1-0.3',
                standard: '0.1',
                large_data: '0.01-0.1'
            }
        },
        {
            name: 'Warmup Steps',
            key: 'warmup_steps',
            category: 'training',
            type: 'slider',
            min: 0,
            max: 10000,
            default: 2000,
            step: 100,
            description: 'Number of steps to linearly increase learning rate from 0. Prevents early training instability from large gradient updates.',
            tip: '1-5% of total steps; critical for stability',
            compute: 'none',
            impact: 'high',
            tradeoffs: {
                increase: ['More stable start', 'Better for large LR', 'Slower initial progress'],
                decrease: ['Faster warmup', 'Risk of instability', 'May diverge']
            },
            research: 'Vaswani et al. (2017) used warmup for original Transformer. Most LLMs use 1-5% of total training steps.',
            recommendations: {
                short_training: '100-500 steps',
                standard: '2000-4000 steps',
                large_models: '5000-10000 steps'
            }
        },
        {
            name: 'Gradient Checkpointing',
            key: 'gradient_checkpointing',
            category: 'efficiency',
            type: 'toggle',
            default: false,
            description: 'Trade compute for memory by recomputing activations during backward pass instead of storing them. Reduces memory ~60% at ~20% speed cost.',
            tip: 'Enable when memory-constrained; 20% slower',
            compute: 'medium',
            impact: 'high',
            tradeoffs: {
                enabled: ['~60% less activation memory', 'Train larger models', '~20% slower training'],
                disabled: ['Faster training', 'More memory required', 'Standard for small models']
            },
            research: 'Chen et al. (2016) introduced gradient checkpointing. Essential for training large models on limited GPU memory.',
            recommendations: {
                plenty_memory: 'Disable',
                memory_constrained: 'Enable',
                very_large_models: 'Enable with selective checkpointing'
            }
        },
        {
            name: 'Learning Rate',
            key: 'learning_rate',
            category: 'training',
            type: 'slider',
            min: 0.00001,
            max: 0.001,
            default: 0.0003,
            step: 0.00001,
            description: 'Peak learning rate for the optimizer. Typically follows warmup then cosine decay schedule.',
            tip: '3e-4 common; scale down for larger models',
            compute: 'none',
            impact: 'high',
            tradeoffs: {
                increase: ['Faster learning', 'May overshoot optima', 'Risk of instability'],
                decrease: ['More stable', 'Slower convergence', 'May underfit']
            },
            research: 'Scaling laws suggest LR should decrease as model size increases. GPT-3 175B uses 6e-5, while smaller models use 3e-4.',
            recommendations: {
                small: '1e-4 to 3e-4',
                medium: '1e-4 to 2e-4',
                large: '3e-5 to 1e-4'
            }
        }
    ]
};
