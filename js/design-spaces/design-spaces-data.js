/**
 * Design Spaces Parameter Data
 * Comprehensive reference for architecture design parameters
 * Enhanced with research-backed information, categories, and scaling guidelines
 */

// Parameter categories for organization
export const parameterCategories = {
    architecture: { name: 'Architecture', icon: '🏗️', description: 'Core structural decisions' },
    attention: { name: 'Attention', icon: '👁️', description: 'Attention mechanism configuration' },
    training: { name: 'Training', icon: '⚙️', description: 'Training dynamics and optimization' },
    regularization: { name: 'Regularization', icon: '🛡️', description: 'Preventing overfitting' },
    efficiency: { name: 'Efficiency', icon: '⚡', description: 'Compute and memory optimization' },
    generation: { name: 'Generation', icon: '✨', description: 'Output generation settings' }
};

export const architectures = [
    {
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
            }
        ]
    },
    {
        id: 'llms',
        name: 'Large Language Models',
        icon: '🗣️',
        color: '#8b5cf6',
        description: 'Scaled transformer models for text generation',
        fullDescription: 'LLMs are decoder-only transformers trained autoregressively on massive text corpora. Scale unlocks emergent capabilities like in-context learning, chain-of-thought reasoning, and instruction following.',
        whenToUse: 'Text generation, chat, code, reasoning tasks',
        innovation: 'Scaling laws - performance predictable from compute budget',
        typicalSize: '1B - 405B parameters',
        yearIntroduced: 2018,
        keyPapers: [
            { title: 'Training Compute-Optimal LLMs (Chinchilla)', authors: 'Hoffmann et al.', year: 2022, arxiv: '2203.15556' },
            { title: 'LLaMA: Open Foundation Models', authors: 'Touvron et al.', year: 2023, arxiv: '2302.13971' },
            { title: 'Scaling Laws for Neural Language Models', authors: 'Kaplan et al.', year: 2020, arxiv: '2001.08361' }
        ],
        scalingGuidelines: {
            small: { params: '1-3B', tokens: '50-100B', note: 'Good for specific domains, edge deployment' },
            medium: { params: '7-13B', tokens: '1-2T', note: 'Strong general capabilities, fits single GPU' },
            large: { params: '34-70B+', tokens: '2-15T', note: 'Near frontier capabilities, requires multi-GPU' }
        },
        parameters: [
            {
                name: 'Total Params',
                key: 'total_params',
                category: 'architecture',
                type: 'dropdown',
                options: ['1B', '3B', '7B', '13B', '34B', '70B', '405B'],
                default: '7B',
                description: 'Total model parameters. The Chinchilla scaling law suggests training tokens should be ~20x parameters for compute-optimal training, but inference-optimal models train longer.',
                tip: 'Follow Chinchilla for training; overtrain for inference',
                compute: 'varies',
                impact: 'high',
                tradeoffs: {
                    increase: ['Better capabilities', 'More emergent abilities', 'Higher costs'],
                    decrease: ['Faster inference', 'Lower serving costs', 'Easier deployment']
                },
                research: 'Chinchilla showed 70B model with 1.4T tokens beats 280B with 300B tokens. LLaMA trained 7B on 1T tokens (5x overtrained) for better inference efficiency.',
                recommendations: {
                    edge: '1-3B for mobile/edge',
                    standard: '7-13B for most applications',
                    frontier: '70B+ for maximum capability'
                }
            },
            {
                name: 'Context Length',
                key: 'context_length',
                category: 'architecture',
                type: 'slider',
                min: 512,
                max: 128000,
                default: 4096,
                step: 512,
                description: 'Maximum sequence length the model can process. Longer contexts enable more information but cost O(n²) in attention memory and compute.',
                tip: 'Longer = more memory; use sparse attention for 100k+',
                compute: 'high',
                impact: 'high',
                tradeoffs: {
                    increase: ['More context for reasoning', 'Full document processing', 'Better RAG'],
                    decrease: ['Faster inference', 'Much less memory', 'Lower latency']
                },
                research: 'Claude and GPT-4 use 100k+ contexts. Techniques like RoPE scaling, ALiBi, and ring attention enable long contexts.',
                recommendations: {
                    chat: '4k-8k sufficient for most chat',
                    documents: '32k-128k for document processing',
                    coding: '8k-32k for code generation'
                }
            },
            {
                name: 'Vocab Size',
                key: 'vocab_size',
                category: 'architecture',
                type: 'slider',
                min: 32000,
                max: 256000,
                default: 50000,
                step: 1000,
                description: 'Number of tokens in the vocabulary. Larger vocabularies reduce sequence length but increase embedding table size. Multilingual models need larger vocabs.',
                tip: 'Larger for multilingual; 32k-50k for English-only',
                compute: 'medium',
                impact: 'medium',
                tradeoffs: {
                    increase: ['Better multilingual', 'Shorter sequences', 'Larger embedding tables'],
                    decrease: ['Smaller model', 'Longer sequences', 'English-optimized']
                },
                research: 'GPT-4 uses ~100k tokens. LLaMA uses 32k. Larger vocabs help with code (preserving whitespace) and math.',
                recommendations: {
                    english: '32k-50k',
                    multilingual: '100k-150k',
                    code_heavy: '50k-100k'
                }
            },
            {
                name: 'Tokenizer',
                key: 'tokenizer',
                category: 'architecture',
                type: 'dropdown',
                options: ['BPE', 'SentencePiece', 'Tiktoken', 'Unigram'],
                default: 'BPE',
                description: 'Algorithm for splitting text into tokens. BPE with byte-fallback handles any Unicode. SentencePiece is language-agnostic.',
                tip: 'BPE with byte fallback is robust',
                compute: 'none',
                impact: 'medium',
                tradeoffs: {
                    BPE: ['Most common', 'Byte-fallback handles all text', 'GPT-style'],
                    SentencePiece: ['Language agnostic', 'Good for multilingual', 'LLaMA uses this'],
                    Tiktoken: ['Fast implementation', 'OpenAI standard', 'Production optimized'],
                    Unigram: ['Probabilistic', 'Good compression', 'Less common']
                },
                research: 'Kudo & Richardson (2018) introduced SentencePiece. Modern LLMs mostly use BPE variants.',
                recommendations: {
                    default: 'BPE with byte-fallback',
                    multilingual: 'SentencePiece',
                    production: 'Tiktoken for speed'
                }
            },
            {
                name: 'Architecture',
                key: 'architecture',
                category: 'efficiency',
                type: 'dropdown',
                options: ['Dense', 'MoE'],
                default: 'Dense',
                description: 'Dense uses all parameters for every token. Mixture of Experts (MoE) routes tokens to subset of "expert" FFNs, enabling larger models at same inference cost.',
                tip: 'MoE for >70B params; ~3x params for same compute',
                compute: 'varies',
                impact: 'high',
                tradeoffs: {
                    Dense: ['Simple', 'Predictable', 'No routing overhead', 'Standard choice'],
                    MoE: ['3x params/compute', 'Harder to train', 'Load balancing needed', 'Better at scale']
                },
                research: 'Mixtral 8x7B matches 70B dense quality at 7B inference cost. Switch Transformer (Fedus et al., 2021) showed trillion-param MoE feasibility.',
                recommendations: {
                    small: 'Dense',
                    medium: 'Dense (MoE overhead not worth it)',
                    large: 'MoE for 70B+ equivalent capability'
                }
            },
            {
                name: 'Expert Count (MoE)',
                key: 'num_experts',
                category: 'efficiency',
                type: 'slider',
                min: 8,
                max: 128,
                default: 8,
                step: 8,
                description: 'Number of expert FFNs in MoE layers. More experts = more total parameters but same active parameters. Requires load balancing.',
                tip: '8 experts is common; 64+ for very large models',
                compute: 'medium',
                impact: 'high',
                tradeoffs: {
                    increase: ['More total capacity', 'Better specialization', 'Harder load balancing'],
                    decrease: ['Simpler routing', 'Better expert utilization', 'Less communication']
                },
                research: 'Mixtral uses 8 experts with top-2 routing. GShard and Switch explored 64-2048 experts.',
                recommendations: {
                    standard: '8 experts',
                    large: '16-64 experts',
                    research: '64+ for trillion-param models'
                }
            },
            {
                name: 'Top-K Routing',
                key: 'top_k',
                category: 'efficiency',
                type: 'slider',
                min: 1,
                max: 4,
                default: 2,
                step: 1,
                description: 'How many experts process each token. Top-1 is most efficient, Top-2 adds redundancy and smoother gradients.',
                tip: '2 experts per token standard for quality',
                compute: 'low',
                impact: 'medium',
                tradeoffs: {
                    increase: ['Better quality', 'More robust routing', 'Higher compute'],
                    decrease: ['Faster inference', 'Cleaner specialization', 'May hurt quality']
                },
                research: 'GShard used Top-2. Some recent work explores Top-1 with better auxiliary losses.',
                recommendations: {
                    efficient: 'Top-1 for speed',
                    balanced: 'Top-2 (standard)',
                    quality: 'Top-2 or Top-4'
                }
            },
            {
                name: 'KV Cache',
                key: 'kv_cache',
                category: 'efficiency',
                type: 'dropdown',
                options: ['Full', 'GQA', 'MQA', 'Paged', 'Sliding Window'],
                default: 'GQA',
                description: 'Key-Value cache strategy for inference. Full MHA caches all heads. GQA/MQA share KV across query heads. Paged manages memory dynamically.',
                tip: 'GQA reduces memory 4-8x with minimal quality loss',
                compute: 'varies',
                impact: 'medium',
                tradeoffs: {
                    Full: ['Maximum quality', 'Highest memory', 'Standard baseline'],
                    GQA: ['Good quality/memory balance', '4-8x memory reduction', 'LLaMA-2 standard'],
                    MQA: ['Minimum memory', 'Some quality loss', 'Best for very long contexts'],
                    Paged: ['Dynamic allocation', 'vLLM standard', 'Best utilization'],
                    'Sliding Window': ['Fixed memory', 'Local attention only', 'Mistral approach']
                },
                research: 'vLLM introduced paged attention for 24x throughput. Mistral uses sliding window (4k) with full attention at certain layers.',
                recommendations: {
                    training: 'Full (GQA architecture)',
                    serving: 'Paged + GQA',
                    long_context: 'Sliding Window + sparse'
                }
            },
            {
                name: 'Tie Embeddings',
                key: 'tie_embeddings',
                category: 'efficiency',
                type: 'toggle',
                default: true,
                description: 'Share weights between input embeddings and output projection. Reduces parameters by vocab_size × d_model (~150M for typical LLM).',
                tip: 'Saves ~10-15% params with minimal quality impact',
                compute: 'none',
                impact: 'low',
                tradeoffs: {
                    tied: ['Fewer parameters', 'Slight regularization', 'Standard practice'],
                    untied: ['More capacity', 'Separate input/output spaces', 'Marginal gains']
                },
                research: 'Press & Wolf (2016) showed tying works well. Most modern LLMs tie embeddings.',
                recommendations: {
                    default: 'Tie embeddings (standard)',
                    experimental: 'Untie for very large vocab'
                }
            },
            {
                name: 'RoPE Base',
                key: 'rope_base',
                category: 'architecture',
                type: 'slider',
                min: 10000,
                max: 1000000,
                default: 10000,
                step: 10000,
                description: 'Base frequency for Rotary Position Embeddings. Higher values enable longer context extrapolation. Can be increased post-training.',
                tip: 'Increase for longer contexts; 500k for 100k+ context',
                compute: 'none',
                impact: 'medium',
                tradeoffs: {
                    increase: ['Better length extrapolation', 'Longer effective context', 'May need fine-tuning'],
                    decrease: ['Better short-range precision', 'Original training distribution', 'Standard choice']
                },
                research: 'Code Llama extended to 100k context by scaling RoPE base to 1M. YaRN (Peng et al., 2023) proposes better scaling methods.',
                recommendations: {
                    '4k': '10000 (default)',
                    '32k': '100000-500000',
                    '100k+': '500000-1000000'
                }
            }
        ]
    },
    {
        id: 'diffusion',
        name: 'Diffusion Models',
        icon: '🌊',
        color: '#ec4899',
        description: 'Denoising-based generative models',
        fullDescription: 'Diffusion models learn to reverse a gradual noising process, generating samples by iterative denoising. They achieve state-of-the-art image quality and have been extended to video, audio, and 3D generation.',
        whenToUse: 'Image generation, audio synthesis, video generation',
        innovation: 'Learn to reverse a gradual noising process',
        typicalSize: '100M - 2B parameters',
        yearIntroduced: 2020,
        keyPapers: [
            { title: 'Denoising Diffusion Probabilistic Models', authors: 'Ho et al.', year: 2020, arxiv: '2006.11239' },
            { title: 'High-Resolution Image Synthesis with Latent Diffusion', authors: 'Rombach et al.', year: 2022, arxiv: '2112.10752' },
            { title: 'Scalable Diffusion Models with Transformers (DiT)', authors: 'Peebles & Xie', year: 2023, arxiv: '2212.09748' }
        ],
        scalingGuidelines: {
            small: { params: '<500M', steps: '20-50', note: 'Fast generation, good for prototyping' },
            medium: { params: '500M-1B', steps: '20-30', note: 'Production quality, Stable Diffusion range' },
            large: { params: '>1B', steps: '20-50', note: 'Frontier quality, DALL-E 3 / Midjourney range' }
        },
        parameters: [
            {
                name: 'Timesteps (T)',
                key: 'timesteps',
                category: 'architecture',
                type: 'slider',
                min: 100,
                max: 1000,
                default: 1000,
                description: 'Total diffusion steps during training. More steps = finer noise schedule but slower. Inference can use far fewer steps with good samplers.',
                tip: '1000 for training; 20-50 for inference with DDIM/DPM++',
                compute: 'high',
                impact: 'high',
                tradeoffs: {
                    increase: ['Finer noise schedule', 'Better gradients', 'Slower training'],
                    decrease: ['Faster training', 'Coarser schedule', 'May hurt quality']
                },
                research: 'DDPM used 1000 steps. Modern samplers (DDIM, DPM++) achieve similar quality in 20-50 steps.',
                recommendations: {
                    training: '1000 steps',
                    fast_inference: '20-30 steps with DPM++',
                    quality_inference: '50 steps'
                }
            },
            {
                name: 'Noise Schedule',
                key: 'noise_schedule',
                category: 'training',
                type: 'dropdown',
                options: ['Linear', 'Cosine', 'Scaled Linear', 'Sigmoid'],
                default: 'Cosine',
                description: 'How noise is added across timesteps. Cosine preserves more signal in early steps, improving detail retention. Scaled linear is SD default.',
                tip: 'Cosine preserves more signal; scaled linear for SD',
                compute: 'none',
                impact: 'high',
                tradeoffs: {
                    Linear: ['Simple', 'Original DDPM', 'Loses details early'],
                    Cosine: ['Better detail preservation', 'Improved Text2Image', 'Most versatile'],
                    'Scaled Linear': ['SD default', 'Good balance', 'Tuned for latent space'],
                    Sigmoid: ['Smooth transitions', 'Recent research', 'May improve consistency']
                },
                research: 'Nichol & Dhariwal (2021) showed cosine schedule improves FID. SD uses scaled linear (β_start=0.00085, β_end=0.012).',
                recommendations: {
                    default: 'Cosine',
                    latent_diffusion: 'Scaled Linear',
                    experimental: 'Sigmoid or learned'
                }
            },
            {
                name: 'Beta Start',
                key: 'beta_start',
                category: 'training',
                type: 'slider',
                min: 0.0001,
                max: 0.001,
                default: 0.0001,
                step: 0.0001,
                description: 'Initial noise variance. Lower values preserve more signal in early steps, helping the model learn fine details.',
                tip: 'Lower = more detail preservation in early steps',
                compute: 'none',
                impact: 'medium',
                tradeoffs: {
                    increase: ['Faster destruction', 'Quicker to noise'],
                    decrease: ['More detail preservation', 'Better fine features']
                },
                research: 'SD uses 0.00085. DDPM used 0.0001. Very sensitive parameter.',
                recommendations: {
                    default: '0.0001',
                    SD_style: '0.00085',
                    high_detail: '0.00005'
                }
            },
            {
                name: 'Beta End',
                key: 'beta_end',
                category: 'training',
                type: 'slider',
                min: 0.01,
                max: 0.1,
                default: 0.02,
                step: 0.01,
                description: 'Final noise variance. Higher values ensure complete noise at final timestep, but too high can make early generation difficult.',
                tip: 'Higher = more noise at end; 0.02 is typical',
                compute: 'none',
                impact: 'medium',
                tradeoffs: {
                    increase: ['More complete destruction', 'Better mode coverage'],
                    decrease: ['Easier early generation', 'May not reach pure noise']
                },
                research: 'DDPM used 0.02. SD uses 0.012 (scaled linear). Should reach near-pure noise.',
                recommendations: {
                    default: '0.02',
                    SD_style: '0.012',
                    careful: 'Verify signal destroyed at T'
                }
            },
            {
                name: 'Sampler',
                key: 'sampler',
                category: 'generation',
                type: 'dropdown',
                options: ['DDPM', 'DDIM', 'Euler', 'DPM++', 'DPM++ 2M', 'Heun', 'UniPC'],
                default: 'DPM++ 2M',
                description: 'Algorithm for iterative denoising during inference. Advanced samplers achieve DDPM quality in 10-50x fewer steps.',
                tip: 'DPM++ 2M for quality/speed balance; Euler for speed',
                compute: 'varies',
                impact: 'high',
                tradeoffs: {
                    DDPM: ['Original', 'Slow (1000 steps)', 'Baseline quality'],
                    DDIM: ['Deterministic', '50-100 steps', 'Good for interpolation'],
                    Euler: ['Fast', '20-30 steps', 'Simple, effective'],
                    'DPM++': ['State-of-art', '20-30 steps', 'Best quality/speed'],
                    'DPM++ 2M': ['Multi-step', '20-30 steps', 'Most recommended'],
                    Heun: ['Second-order', '30-50 steps', 'Very accurate'],
                    UniPC: ['Universal', '10-20 steps', 'Fastest quality']
                },
                research: 'Lu et al. (2022) introduced DPM++. UniPC (Zhao et al., 2023) further reduces steps.',
                recommendations: {
                    fast: 'Euler or UniPC (10-20 steps)',
                    balanced: 'DPM++ 2M (20-30 steps)',
                    quality: 'Heun (30-50 steps)'
                }
            },
            {
                name: 'Inference Steps',
                key: 'inference_steps',
                category: 'generation',
                type: 'slider',
                min: 10,
                max: 100,
                default: 20,
                step: 5,
                description: 'Number of denoising steps during generation. More steps = higher quality but slower. Sweet spot depends on sampler choice.',
                tip: '20-30 for most samplers; 50+ for maximum quality',
                compute: 'high',
                impact: 'high',
                tradeoffs: {
                    increase: ['Better quality', 'More detail', 'Slower generation'],
                    decrease: ['Faster generation', 'May lose detail', 'Good enough for previews']
                },
                research: 'With DPM++, 20-30 steps matches 1000-step DDPM quality. Diminishing returns above 50.',
                recommendations: {
                    preview: '10-15 steps',
                    production: '20-30 steps',
                    maximum_quality: '50 steps'
                }
            },
            {
                name: 'Guidance Scale',
                key: 'guidance_scale',
                category: 'generation',
                type: 'slider',
                min: 1,
                max: 20,
                default: 7.5,
                step: 0.5,
                description: 'Classifier-free guidance strength. Higher values follow the prompt more closely but can reduce diversity and cause artifacts.',
                tip: '7-9 typical; higher for strict prompt following',
                compute: 'low',
                impact: 'high',
                tradeoffs: {
                    increase: ['Stronger prompt adherence', 'More contrast', 'Can cause artifacts'],
                    decrease: ['More diversity', 'Softer outputs', 'May ignore prompt details']
                },
                research: 'Ho & Salimans (2022) introduced classifier-free guidance. SD default is 7.5.',
                recommendations: {
                    creative: '5-7',
                    balanced: '7.5-9',
                    strict: '10-15'
                }
            },
            {
                name: 'Architecture',
                key: 'architecture_type',
                category: 'architecture',
                type: 'dropdown',
                options: ['U-Net', 'DiT', 'U-ViT'],
                default: 'U-Net',
                description: 'Backbone architecture. U-Net is CNN-based with skip connections. DiT uses transformers. U-ViT combines both.',
                tip: 'U-Net proven; DiT scales better for very large models',
                compute: 'varies',
                impact: 'high',
                tradeoffs: {
                    'U-Net': ['Proven', 'Efficient', 'Good inductive bias', 'SD standard'],
                    DiT: ['Transformer-based', 'Scales better', 'Used by SORA', 'More compute'],
                    'U-ViT': ['Hybrid', 'Best of both', 'Emerging research']
                },
                research: 'DiT (Peebles & Xie, 2023) showed transformers match/beat U-Net at scale. SORA uses DiT.',
                recommendations: {
                    standard: 'U-Net',
                    large_scale: 'DiT',
                    research: 'U-ViT'
                }
            },
            {
                name: 'Attention Resolutions',
                key: 'attention_res',
                category: 'attention',
                type: 'dropdown',
                options: ['32, 16, 8', '16, 8', '8 only', '64, 32, 16, 8'],
                default: '16, 8',
                description: 'At which spatial resolutions to apply self-attention. Attention at higher res is expensive but captures global structure better.',
                tip: 'Lower res = more global attention; balance compute',
                compute: 'high',
                impact: 'medium',
                tradeoffs: {
                    'more_levels': ['Better global coherence', 'Much more compute', 'For high-res'],
                    'fewer_levels': ['Faster', 'Less memory', 'May lose global structure']
                },
                research: 'SD uses attention at 32x32, 16x16, 8x8. Adding 64x64 improves but costs 4x.',
                recommendations: {
                    efficient: '8 only',
                    balanced: '16, 8',
                    quality: '32, 16, 8'
                }
            },
            {
                name: 'Conditioning',
                key: 'conditioning',
                category: 'generation',
                type: 'dropdown',
                options: ['None', 'Class', 'Text (CLIP)', 'Text (T5)', 'Image'],
                default: 'Text (CLIP)',
                description: 'What information guides generation. Text conditioning uses frozen text encoders (CLIP or T5). Image conditioning enables img2img.',
                tip: 'CLIP for general text; T5 for detailed prompts',
                compute: 'medium',
                impact: 'high',
                tradeoffs: {
                    None: ['Unconditional', 'Random generation', 'No control'],
                    Class: ['Simple labels', 'Fast', 'Limited control'],
                    'Text (CLIP)': ['Good prompt following', 'Standard', 'Limited length'],
                    'Text (T5)': ['Long, detailed prompts', 'Better text rendering', 'Imagen style'],
                    Image: ['Image-to-image', 'Style transfer', 'Inpainting']
                },
                research: 'CLIP (Radford et al., 2021) enables open-vocab text. T5 (Saharia et al., 2022 - Imagen) improves text understanding.',
                recommendations: {
                    simple: 'Class conditioning',
                    standard: 'Text (CLIP)',
                    detailed_prompts: 'Text (T5)',
                    editing: 'Image + Text'
                }
            }
        ]
    },
    {
        id: 'gnns',
        name: 'Graph Neural Networks',
        icon: '🕸️',
        color: '#10b981',
        description: 'Neural networks for graph-structured data',
        fullDescription: 'GNNs learn representations by aggregating information from node neighborhoods through message passing. They handle irregular structures like molecules, social networks, and knowledge graphs where standard neural networks fail.',
        whenToUse: 'Molecules, social networks, knowledge graphs, recommendations',
        innovation: 'Message passing between nodes aggregates neighborhood info',
        typicalSize: '100K - 10M parameters',
        yearIntroduced: 2016,
        keyPapers: [
            { title: 'Semi-Supervised Classification with GCNs', authors: 'Kipf & Welling', year: 2017, arxiv: '1609.02907' },
            { title: 'Graph Attention Networks', authors: 'Veličković et al.', year: 2018, arxiv: '1710.10903' },
            { title: 'How Powerful are Graph Neural Networks?', authors: 'Xu et al.', year: 2019, arxiv: '1810.00826' }
        ],
        scalingGuidelines: {
            small: { params: '<1M', layers: '2-3', note: 'Most graph tasks; avoid over-smoothing' },
            medium: { params: '1-10M', layers: '3-5', note: 'Complex graphs with skip connections' },
            large: { params: '>10M', layers: '5+', note: 'Graph transformers for large-scale' }
        },
        parameters: [
            {
                name: 'Layers',
                key: 'layers',
                category: 'architecture',
                type: 'slider',
                min: 2,
                max: 8,
                default: 3,
                step: 1,
                description: 'Number of message passing layers. Each layer expands the receptive field by one hop. Deep GNNs suffer from over-smoothing where node features converge.',
                tip: '2-3 to avoid over-smoothing; use skip connections for deeper',
                compute: 'medium',
                impact: 'high',
                tradeoffs: {
                    increase: ['Larger receptive field', 'More hops considered', 'Risk of over-smoothing'],
                    decrease: ['Local focus only', 'Faster training', 'Less expressive']
                },
                research: 'Li et al. (2018) identified over-smoothing. JK-Net and PairNorm help for deeper networks.',
                recommendations: {
                    node_classification: '2-3 layers',
                    graph_classification: '3-5 layers with pooling',
                    deep: 'Use skip connections + PairNorm'
                }
            },
            {
                name: 'Hidden Dim',
                key: 'hidden_dim',
                category: 'architecture',
                type: 'slider',
                min: 32,
                max: 512,
                default: 128,
                step: 32,
                description: 'Dimension of node representations. Larger dimensions capture more complex features but risk overfitting on small graphs.',
                tip: 'Depends on graph complexity; 64-256 typical',
                compute: 'medium',
                impact: 'medium',
                tradeoffs: {
                    increase: ['Richer representations', 'Better for complex patterns', 'More overfitting risk'],
                    decrease: ['Faster', 'Better generalization', 'May underfit']
                },
                research: 'Most benchmarks use 64-256. Molecular tasks often need larger dims.',
                recommendations: {
                    small_graphs: '64-128',
                    citation_networks: '128-256',
                    molecules: '256-512'
                }
            },
            {
                name: 'Aggregation',
                key: 'aggregation',
                category: 'architecture',
                type: 'dropdown',
                options: ['Mean', 'Sum', 'Max', 'Attention', 'PNA'],
                default: 'Mean',
                description: 'How to combine neighbor messages. Sum preserves injective property (distinguishes different multisets). Mean normalizes by degree. PNA uses multiple.',
                tip: 'Sum for counting; Mean for averaging; PNA for expressivity',
                compute: 'low',
                impact: 'high',
                tradeoffs: {
                    Mean: ['Degree-normalized', 'Good default', 'May lose count info'],
                    Sum: ['Preserves counts', 'More expressive', 'Sensitive to degree'],
                    Max: ['Highlights important', 'Loses info', 'Good for some tasks'],
                    Attention: ['Learnable weights', 'GAT style', 'More parameters'],
                    PNA: ['Combines all', 'Most expressive', 'Highest compute']
                },
                research: 'Xu et al. (2019) proved sum is most expressive for GIN. Corso et al. (2020) introduced PNA with multiple aggregators.',
                recommendations: {
                    default: 'Mean',
                    maximum_expressivity: 'Sum (GIN) or PNA',
                    heterogeneous_graphs: 'Attention'
                }
            },
            {
                name: 'Message Passing',
                key: 'message_passing',
                category: 'architecture',
                type: 'dropdown',
                options: ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'MPNN'],
                default: 'GCN',
                description: 'Message passing scheme. GCN uses spectral convolution. GAT uses attention. GIN is maximally expressive under WL test.',
                tip: 'GIN for expressivity; GAT for heterogeneous; GCN for speed',
                compute: 'varies',
                impact: 'high',
                tradeoffs: {
                    GCN: ['Fast', 'Spectral basis', 'Good baseline', 'Limited expressivity'],
                    GAT: ['Attention weights', 'Edge-aware', 'More parameters'],
                    GraphSAGE: ['Inductive', 'Sampling-friendly', 'Scalable'],
                    GIN: ['WL-equivalent', 'Most expressive', 'Good for graph-level'],
                    MPNN: ['General framework', 'Customizable', 'Flexible']
                },
                research: 'Xu et al. (2019) showed GIN matches 1-WL test. GAT (Veličković et al., 2018) adds learnable edge weights.',
                recommendations: {
                    node_tasks: 'GCN or GAT',
                    graph_tasks: 'GIN',
                    large_scale: 'GraphSAGE',
                    molecules: 'MPNN'
                }
            },
            {
                name: 'Skip Connections',
                key: 'skip_connections',
                category: 'architecture',
                type: 'dropdown',
                options: ['None', 'Residual', 'Dense', 'JumpingKnowledge'],
                default: 'Residual',
                description: 'How to connect across layers. Skip connections are essential for deeper GNNs to avoid over-smoothing and enable gradient flow.',
                tip: 'JK for variable receptive fields; Residual for stability',
                compute: 'low',
                impact: 'high',
                tradeoffs: {
                    None: ['Simplest', 'Limited depth', 'Over-smoothing prone'],
                    Residual: ['Standard', 'Enables depth', 'Fixed receptive field'],
                    Dense: ['All layers connected', 'Heavy memory', 'DenseNet style'],
                    JumpingKnowledge: ['Adaptive receptive field', 'Combines all layers', 'Best for varied graphs']
                },
                research: 'Xu et al. (2018) introduced Jumping Knowledge for adaptive aggregation of all layer outputs.',
                recommendations: {
                    shallow: 'None or Residual',
                    deep: 'JumpingKnowledge',
                    varied_graphs: 'JumpingKnowledge'
                }
            },
            {
                name: 'Normalization',
                key: 'normalization',
                category: 'training',
                type: 'dropdown',
                options: ['None', 'BatchNorm', 'LayerNorm', 'GraphNorm', 'PairNorm'],
                default: 'BatchNorm',
                description: 'Normalization technique. PairNorm specifically prevents over-smoothing by keeping representations differentiated across nodes.',
                tip: 'PairNorm prevents over-smoothing; GraphNorm for batches',
                compute: 'low',
                impact: 'medium',
                tradeoffs: {
                    None: ['Simplest', 'May be unstable'],
                    BatchNorm: ['Standard', 'Requires batching'],
                    LayerNorm: ['Per-node', 'Graph-size independent'],
                    GraphNorm: ['Graph-aware', 'Learnable interpolation'],
                    PairNorm: ['Anti-over-smoothing', 'Keeps nodes distinct']
                },
                research: 'PairNorm (Zhao & Akoglu, 2020) specifically addresses over-smoothing. GraphNorm (Cai et al., 2021) learns batch/layer tradeoff.',
                recommendations: {
                    default: 'BatchNorm',
                    deep_gnns: 'PairNorm',
                    variable_size: 'LayerNorm or GraphNorm'
                }
            },
            {
                name: 'Dropout',
                key: 'dropout',
                category: 'regularization',
                type: 'slider',
                min: 0,
                max: 0.5,
                default: 0.5,
                step: 0.1,
                description: 'Node feature dropout rate. GNNs often need higher dropout than other architectures due to message passing redundancy.',
                tip: 'GNNs often need higher dropout (0.3-0.5)',
                compute: 'none',
                impact: 'medium',
                tradeoffs: {
                    increase: ['Stronger regularization', 'Less overfitting'],
                    decrease: ['Faster convergence', 'Better for large graphs']
                },
                research: 'Standard practice is 0.5 for node classification on citation networks.',
                recommendations: {
                    small_graphs: '0.5',
                    large_graphs: '0.2-0.3',
                    molecules: '0.0-0.2'
                }
            },
            {
                name: 'Edge Dropout',
                key: 'edge_dropout',
                category: 'regularization',
                type: 'slider',
                min: 0,
                max: 0.5,
                default: 0.2,
                step: 0.1,
                description: 'Randomly drop edges during training. DropEdge augmentation helps generalization and reduces over-smoothing.',
                tip: 'DropEdge helps generalization and reduces over-smoothing',
                compute: 'none',
                impact: 'medium',
                tradeoffs: {
                    increase: ['More regularization', 'Graph augmentation effect'],
                    decrease: ['Preserve graph structure', 'Faster']
                },
                research: 'Rong et al. (2020) showed DropEdge helps deeper GNNs by reducing message passing rate.',
                recommendations: {
                    default: '0.2',
                    deep_gnns: '0.3-0.5',
                    dense_graphs: '0.2-0.3'
                }
            },
            {
                name: 'Readout',
                key: 'readout',
                category: 'architecture',
                type: 'dropdown',
                options: ['Mean', 'Sum', 'Max', 'Set2Set', 'Virtual Node', 'Attention'],
                default: 'Mean',
                description: 'How to get graph-level representation from node embeddings. Sum is more expressive. Virtual node enables global communication.',
                tip: 'Virtual node for global context; attention for weighted',
                compute: 'low',
                impact: 'medium',
                tradeoffs: {
                    Mean: ['Size-invariant', 'Simple', 'Good default'],
                    Sum: ['Size-sensitive', 'More expressive', 'May need normalization'],
                    Max: ['Highlights key nodes', 'Loses info'],
                    Set2Set: ['Learnable', 'Order-invariant', 'LSTM-based'],
                    'Virtual Node': ['Global communication', 'Each layer', 'OGB recommended'],
                    Attention: ['Weighted readout', 'Highlights important nodes']
                },
                research: 'Gilmer et al. (2017) used Set2Set. Virtual node (OGB) adds pseudo-node connected to all.',
                recommendations: {
                    default: 'Mean',
                    graph_classification: 'Sum or Virtual Node',
                    molecules: 'Attention or Set2Set'
                }
            },
            {
                name: 'Positional Encoding',
                key: 'pos_encoding',
                category: 'architecture',
                type: 'dropdown',
                options: ['None', 'Laplacian', 'Random Walk', 'Learnable', 'SignNet'],
                default: 'Laplacian',
                description: 'Position information for nodes. Without it, GNNs cannot distinguish symmetric structures. Laplacian eigenvectors encode graph structure.',
                tip: 'Laplacian for structure; Random Walk for local position',
                compute: 'medium',
                impact: 'high',
                tradeoffs: {
                    None: ['No position info', 'Fastest', 'Cannot distinguish symmetric nodes'],
                    Laplacian: ['Global structure', 'Eigendecomposition needed', 'Sign ambiguity'],
                    'Random Walk': ['Local structure', 'Fast to compute', 'Less global'],
                    Learnable: ['Task-specific', 'Needs node ordering', 'Not permutation-invariant'],
                    SignNet: ['Sign-invariant Laplacian', 'Addresses sign ambiguity', 'More robust']
                },
                research: 'Dwivedi et al. (2020) showed Laplacian PE helps. SignNet (Lim et al., 2022) handles sign ambiguity.',
                recommendations: {
                    standard: 'None',
                    graph_transformers: 'Laplacian or Random Walk',
                    molecular: 'Random Walk + Laplacian'
                }
            }
        ]
    },
    {
        id: 'cnns',
        name: 'Convolutional Neural Networks',
        icon: '🔲',
        color: '#f59e0b',
        description: 'Spatial feature extractors for images',
        fullDescription: 'CNNs exploit spatial structure through local connectivity and weight sharing. Convolutional layers detect features at multiple scales, while pooling provides translation invariance. Modern CNNs include skip connections and efficient building blocks.',
        whenToUse: 'Image classification, object detection, segmentation',
        innovation: 'Local connectivity and weight sharing exploit spatial structure',
        typicalSize: '1M - 100M parameters',
        yearIntroduced: 1998,
        keyPapers: [
            { title: 'Deep Residual Learning for Image Recognition', authors: 'He et al.', year: 2016, arxiv: '1512.03385' },
            { title: 'EfficientNet: Rethinking Model Scaling', authors: 'Tan & Le', year: 2019, arxiv: '1905.11946' },
            { title: 'Designing Network Design Spaces (RegNet)', authors: 'Radosavovic et al.', year: 2020, arxiv: '2003.13678' }
        ],
        scalingGuidelines: {
            small: { params: '<5M', note: 'MobileNet-style for edge/mobile devices' },
            medium: { params: '5-50M', note: 'ResNet-50 range for general use' },
            large: { params: '>50M', note: 'EfficientNet-L2, ConvNeXt-XL for maximum accuracy' }
        },
        parameters: [
            { name: 'Kernel Size', key: 'kernel_size', category: 'architecture', type: 'dropdown', options: ['3x3', '5x5', '7x7'], default: '3x3', description: 'Size of convolutional filters. Two 3x3 layers have same receptive field as one 5x5 but fewer parameters and more non-linearity.', tip: '3x3 most efficient (VGG insight)', compute: 'varies', impact: 'medium', tradeoffs: { '3x3': ['Most efficient', 'More non-linearity', 'Standard choice'], '5x5': ['Larger receptive field', 'Fewer layers needed', 'More parameters'], '7x7': ['Stem layer only', 'Large initial receptive field', 'ResNet style'] }, research: 'VGGNet showed stacking 3x3 beats larger kernels. ConvNeXt uses 7x7 depthwise convs.', recommendations: { default: '3x3', stem: '7x7 strided', modern: '7x7 depthwise (ConvNeXt)' } },
            { name: 'Depth', key: 'depth', category: 'architecture', type: 'slider', min: 5, max: 152, default: 50, step: 1, description: 'Total number of layers. Deeper networks need skip connections to train. Follow compound scaling for balanced depth/width/resolution.', tip: 'ResNet-50 is versatile baseline; scale with data', compute: 'high', impact: 'high', tradeoffs: { increase: ['More representational power', 'Better features', 'Harder to train'], decrease: ['Faster', 'Easier optimization', 'May underfit'] }, research: 'He et al. showed residual connections enable 152+ layers. RegNet found optimal depth scales as d ∝ log(flops).', recommendations: { mobile: '18-34', standard: '50', maximum: '101-152' } },
            { name: 'Width Multiplier', key: 'width_mult', category: 'architecture', type: 'slider', min: 0.25, max: 2.0, default: 1.0, step: 0.25, description: 'Scales channel counts throughout network. Width is the most compute-efficient dimension to scale according to EfficientNet.', tip: 'Reduce for mobile; increase for accuracy', compute: 'high', impact: 'high', tradeoffs: { increase: ['More capacity', 'Better accuracy', 'Quadratic compute'], decrease: ['Faster inference', 'Mobile-friendly', 'May underfit'] }, research: 'EfficientNet compound scaling: width ∝ α^φ, depth ∝ β^φ, resolution ∝ γ^φ.', recommendations: { mobile: '0.25-0.5', standard: '1.0', high_accuracy: '1.5-2.0' } },
            { name: 'Pooling', key: 'pooling', category: 'architecture', type: 'dropdown', options: ['Max', 'Avg', 'Strided Conv'], default: 'Max', description: 'How to downsample spatial dimensions. Strided convolutions are learnable and increasingly preferred.', tip: 'Strided conv more flexible; max for invariance', compute: 'low', impact: 'medium', tradeoffs: { Max: ['Translation invariance', 'No parameters', 'Discards info'], Avg: ['Smoother', 'No parameters', 'Blurs features'], 'Strided Conv': ['Learnable', 'More parameters', 'Best accuracy'] }, research: 'Springenberg et al. (2014) showed strided convs can replace pooling with better results.', recommendations: { classic: 'Max pooling', modern: 'Strided Conv', efficient: 'Max pooling' } },
            { name: 'Skip Connections', key: 'skip_connections', category: 'architecture', type: 'dropdown', options: ['None', 'Residual', 'Dense'], default: 'Residual', description: 'Connections that bypass layers. Essential for training deep networks by providing gradient highways.', tip: 'Essential for depth > 20; residual is standard', compute: 'low', impact: 'high', tradeoffs: { None: ['Simple', 'Limited depth', 'VGG-style'], Residual: ['Standard', 'Enables deep networks', 'ResNet'], Dense: ['All-to-all', 'Feature reuse', 'Memory heavy'] }, research: 'ResNet showed identity mappings enable very deep networks. DenseNet connects every layer to every other.', recommendations: { shallow: 'None okay', default: 'Residual', maximum_reuse: 'Dense' } },
            { name: 'Normalization', key: 'normalization', category: 'training', type: 'dropdown', options: ['BatchNorm', 'LayerNorm', 'GroupNorm', 'None'], default: 'BatchNorm', description: 'Normalization technique. BatchNorm is standard but requires reasonable batch sizes. GroupNorm works with any batch size.', tip: 'GroupNorm for small batches; BatchNorm otherwise', compute: 'low', impact: 'medium', tradeoffs: { BatchNorm: ['Fast training', 'Batch-dependent', 'Standard'], LayerNorm: ['Batch-independent', 'Good for variable batch'], GroupNorm: ['Small batch friendly', 'Detection/segmentation'], None: ['Simplest', 'Harder to train'] }, research: 'Wu & He (2018) showed GroupNorm matches BatchNorm at batch_size=2 where BN fails.', recommendations: { classification: 'BatchNorm', detection: 'GroupNorm (batch_size varies)', small_batch: 'GroupNorm' } },
            { name: 'Activation', key: 'activation', category: 'architecture', type: 'dropdown', options: ['ReLU', 'LeakyReLU', 'SiLU', 'GELU', 'Mish'], default: 'ReLU', description: 'Non-linearity after convolutions. SiLU (Swish) and GELU are smooth alternatives that often improve accuracy.', tip: 'SiLU for EfficientNet-style; ReLU for speed', compute: 'low', impact: 'low', tradeoffs: { ReLU: ['Fast', 'Simple', 'Can have dead neurons'], LeakyReLU: ['No dead neurons', 'Slightly slower'], SiLU: ['Smooth', 'Better accuracy', 'EfficientNet default'], GELU: ['Smooth', 'Transformer-derived'], Mish: ['Similar to SiLU', 'Slightly different shape'] }, research: 'Ramachandran et al. (2017) found Swish via neural architecture search.', recommendations: { speed: 'ReLU', accuracy: 'SiLU', modern: 'GELU' } },
            { name: 'Stem', key: 'stem', category: 'architecture', type: 'dropdown', options: ['7x7 stride 2', 'Patchify', 'Conv Stack'], default: '7x7 stride 2', description: 'Initial downsampling strategy. Patchify (4x4 non-overlapping conv) is the ConvNeXt approach inspired by ViT.', tip: 'Patchify for ConvNeXt; 7x7 for ResNet', compute: 'low', impact: 'medium', tradeoffs: { '7x7 stride 2': ['Classic ResNet', 'Proven', 'Aggressive downsample'], Patchify: ['ConvNeXt style', 'ViT-inspired', 'Non-overlapping'], 'Conv Stack': ['Gradual downsample', 'More computation', 'Better for small images'] }, research: 'Liu et al. (2022) showed patchify stem is one key to ConvNeXt matching ViT.', recommendations: { classic: '7x7 stride 2', modern: 'Patchify', small_images: 'Conv Stack' } },
            { name: 'SE Blocks', key: 'se_blocks', category: 'architecture', type: 'toggle', default: true, description: 'Squeeze-and-Excitation blocks that recalibrate channel-wise features using global information. Adds ~10% parameters.', tip: 'Squeeze-and-excite adds ~10% params, notable gain', compute: 'low', impact: 'medium', tradeoffs: { enabled: ['Channel attention', 'Better accuracy', 'Small overhead'], disabled: ['Fewer parameters', 'Faster', 'Simpler'] }, research: 'Hu et al. (2018) showed SE blocks give consistent gains across architectures.', recommendations: { default: 'Enable', mobile: 'Enable (efficient gain)', speed_critical: 'Disable' } },
            { name: 'Depthwise Separable', key: 'depthwise', category: 'efficiency', type: 'toggle', default: false, description: 'Split convolution into depthwise (per-channel) and pointwise (1x1) operations. Reduces computation by ~8-9x.', tip: 'Essential for mobile efficiency (MobileNet)', compute: 'low', impact: 'medium', tradeoffs: { enabled: ['8-9x fewer FLOPs', 'Mobile-friendly', 'Slight accuracy drop'], disabled: ['Full convolutions', 'Maximum accuracy', 'More compute'] }, research: 'Howard et al. (2017) MobileNet showed depthwise separable enables mobile deployment.', recommendations: { server: 'Disable', mobile: 'Enable', edge: 'Enable' } }
        ]
    },
    {
        id: 'rnns',
        name: 'RNNs & LSTMs',
        icon: '↩️',
        color: '#ef4444',
        description: 'Recurrent networks for sequential data',
        fullDescription: 'RNNs process sequences by maintaining a hidden state that carries information across time steps. LSTMs and GRUs add gating mechanisms to better capture long-range dependencies and avoid vanishing gradients.',
        whenToUse: 'Time series, speech, when order matters and context is local',
        innovation: 'Hidden state carries information across time steps',
        typicalSize: '100K - 50M parameters',
        yearIntroduced: 1997,
        keyPapers: [
            { title: 'Long Short-Term Memory', authors: 'Hochreiter & Schmidhuber', year: 1997, link: 'Neural Computation' },
            { title: 'Learning Phrase Representations (GRU)', authors: 'Cho et al.', year: 2014, arxiv: '1406.1078' },
            { title: 'Sequence to Sequence Learning', authors: 'Sutskever et al.', year: 2014, arxiv: '1409.3215' }
        ],
        scalingGuidelines: {
            small: { params: '<1M', layers: '1-2', note: 'Simple sequences, real-time' },
            medium: { params: '1-10M', layers: '2-3', note: 'Language modeling, translation' },
            large: { params: '>10M', layers: '4-8', note: 'Complex dependencies (mostly replaced by Transformers)' }
        },
        parameters: [
            { name: 'Hidden Size', key: 'hidden_size', category: 'architecture', type: 'slider', min: 64, max: 2048, default: 512, step: 64, description: 'Dimension of the hidden state. Larger states can memorize more but are slower and more prone to overfitting.', tip: 'Larger = more capacity but slower', compute: 'high', impact: 'high', tradeoffs: { increase: ['More memory capacity', 'Better long-range', 'Slower'], decrease: ['Faster', 'Less overfitting', 'Limited capacity'] }, research: 'State-of-art language models used 1024-2048 before Transformer era.', recommendations: { simple: '128-256', moderate: '512', complex: '1024-2048' } },
            { name: 'Layers', key: 'layers', category: 'architecture', type: 'slider', min: 1, max: 8, default: 2, step: 1, description: 'Number of stacked RNN layers. Deeper RNNs can learn hierarchical representations but are harder to train.', tip: '2-3 typical; deeper needs skip connections', compute: 'high', impact: 'medium', tradeoffs: { increase: ['Hierarchical features', 'More capacity'], decrease: ['Easier training', 'Faster'] }, research: 'Pascanu et al. showed deep RNNs benefit from skip connections like transformers.', recommendations: { default: '2', complex: '3-4', maximum: '4-8 with residual' } },
            { name: 'Cell Type', key: 'cell_type', category: 'architecture', type: 'dropdown', options: ['Vanilla', 'LSTM', 'GRU'], default: 'LSTM', description: 'RNN cell architecture. LSTM has separate cell state and forget gate. GRU is simpler with fewer parameters.', tip: 'GRU simpler but competitive; LSTM for long sequences', compute: 'varies', impact: 'high', tradeoffs: { Vanilla: ['Simple', 'Fast', 'Vanishing gradients'], LSTM: ['Long-range memory', 'More parameters', 'Separate cell state'], GRU: ['Simpler than LSTM', 'Often competitive', 'Fewer gates'] }, research: 'Chung et al. (2014) showed GRU matches LSTM on many tasks with fewer parameters.', recommendations: { default: 'LSTM', efficiency: 'GRU', research: 'Try both' } },
            { name: 'Bidirectional', key: 'bidirectional', category: 'architecture', type: 'toggle', default: true, description: 'Process sequence in both directions. Essential for tasks where future context matters (classification, NER) but not for generation.', tip: 'Yes for classification; No for generation', compute: 'high', impact: 'high', tradeoffs: { enabled: ['Full context', '2x parameters', 'Cannot generate left-to-right'], disabled: ['Causal', 'Generative', 'Only past context'] }, research: 'Schuster & Paliwal (1997) introduced bidirectional RNNs. Essential for BERT-style tasks.', recommendations: { classification: 'Enable', generation: 'Disable', NER: 'Enable' } },
            { name: 'Dropout', key: 'dropout', category: 'regularization', type: 'slider', min: 0, max: 0.5, default: 0.2, step: 0.1, description: 'Dropout between RNN layers (not within recurrence). Standard regularization for RNNs.', tip: 'Applied between layers, not within time steps', compute: 'none', impact: 'medium', tradeoffs: { increase: ['More regularization'], decrease: ['Faster convergence'] }, research: 'Gal & Ghahramani (2016) introduced variational dropout for RNNs.', recommendations: { default: '0.2-0.3', large_data: '0.1', small_data: '0.4-0.5' } },
            { name: 'Recurrent Dropout', key: 'recurrent_dropout', category: 'regularization', type: 'slider', min: 0, max: 0.5, default: 0.0, step: 0.1, description: 'Dropout applied to recurrent connections (within hidden state). Requires variational dropout to work properly.', tip: 'Variational dropout for recurrent connections', compute: 'none', impact: 'medium', tradeoffs: { increase: ['Regularizes hidden dynamics'], decrease: ['Preserve temporal information'] }, research: 'Gal & Ghahramani showed same dropout mask across time steps is essential.', recommendations: { default: '0.0', regularized: '0.1-0.2' } },
            { name: 'Residual', key: 'residual', category: 'architecture', type: 'toggle', default: true, description: 'Add skip connections between RNN layers. Helps gradient flow in deeper networks.', tip: 'Helps deeper networks; essential for 4+ layers', compute: 'low', impact: 'medium', tradeoffs: { enabled: ['Better gradient flow', 'Deeper networks'], disabled: ['Simpler'] }, research: 'Wu et al. showed residual connections help deep RNNs like they help CNNs.', recommendations: { shallow: 'Optional', deep: 'Essential' } },
            { name: 'Layer Norm', key: 'layer_norm', category: 'training', type: 'toggle', default: true, description: 'Apply layer normalization within RNN cells. Stabilizes training especially for longer sequences.', tip: 'Stabilizes training, especially for long sequences', compute: 'low', impact: 'medium', tradeoffs: { enabled: ['Stable training', 'Better with long sequences'], disabled: ['Faster', 'Original design'] }, research: 'Ba et al. (2016) showed LayerNorm helps RNNs more than BatchNorm.', recommendations: { default: 'Enable', speed: 'Disable' } },
            { name: 'Gradient Clipping', key: 'grad_clip', category: 'training', type: 'slider', min: 0.1, max: 10, default: 1.0, step: 0.1, description: 'Maximum gradient norm. Essential for RNN stability due to potential for exploding gradients.', tip: 'Essential for stability; 1.0-5.0 typical', compute: 'none', impact: 'high', tradeoffs: { increase: ['Allow larger updates', 'May be unstable'], decrease: ['More stable', 'Slower learning'] }, research: 'Pascanu et al. (2013) analyzed gradient problems in RNNs. Clipping is standard practice.', recommendations: { default: '1.0', aggressive: '0.25-0.5', relaxed: '5.0' } },
            { name: 'Truncated BPTT', key: 'bptt_len', category: 'training', type: 'slider', min: 35, max: 200, default: 70, step: 5, description: 'Sequence length for backpropagation. Longer captures more context but uses more memory and can cause gradient issues.', tip: 'Longer = more context; balance with memory', compute: 'high', impact: 'medium', tradeoffs: { increase: ['More context', 'More memory', 'Gradient issues'], decrease: ['Less context', 'Faster', 'More stable'] }, research: 'Williams & Peng (1990) introduced TBPTT. Typical values 35-100.', recommendations: { default: '70', long_range: '100-200', efficiency: '35-50' } }
        ]
    },
    {
        id: 'ssms',
        name: 'SSMs / Mamba',
        icon: '〰️',
        color: '#06b6d4',
        description: 'State space models with selective mechanisms',
        fullDescription: 'State Space Models (SSMs) model sequences as continuous dynamical systems, discretized for computation. Mamba adds selectivity - input-dependent state transitions - enabling content-aware processing with linear complexity.',
        whenToUse: 'Long sequences, when linear complexity is needed',
        innovation: 'Selective state spaces - input-dependent dynamics',
        typicalSize: '100M - 8B parameters',
        yearIntroduced: 2023,
        keyPapers: [
            { title: 'Mamba: Linear-Time Sequence Modeling', authors: 'Gu & Dao', year: 2023, arxiv: '2312.00752' },
            { title: 'Efficiently Modeling Long Sequences with S4', authors: 'Gu et al.', year: 2022, arxiv: '2111.00396' },
            { title: 'Hungry Hungry Hippos (H3)', authors: 'Fu et al.', year: 2023, arxiv: '2212.14052' }
        ],
        scalingGuidelines: {
            small: { params: '<500M', note: 'Long-context tasks, efficient inference' },
            medium: { params: '500M-3B', note: 'Competitive with similar-size transformers' },
            large: { params: '>3B', note: 'Emerging frontier, Mamba-2 improvements' }
        },
        parameters: [
            { name: 'Model Dim (D)', key: 'd_model', category: 'architecture', type: 'slider', min: 256, max: 4096, default: 1024, step: 256, description: 'Primary model dimension. Unlike transformers, SSMs scale more linearly with dimension since there is no O(n²) attention.', tip: 'Scale with task complexity; no quadratic cost', compute: 'high', impact: 'high', tradeoffs: { increase: ['More capacity', 'Linear scaling', 'Better representations'], decrease: ['Faster', 'Less memory'] }, research: 'Mamba-370M uses D=1024, Mamba-1.4B uses D=2048. Scales similarly to transformers.', recommendations: { small: '768', medium: '1024-1536', large: '2048-4096' } },
            { name: 'State Expansion (N)', key: 'state_expansion', category: 'architecture', type: 'slider', min: 16, max: 64, default: 16, step: 8, description: 'Dimension of the SSM state per channel. Larger N = more expressivity but more memory. The hidden state has dimension D × N.', tip: 'Larger N = more expressivity; 16 is efficient default', compute: 'high', impact: 'high', tradeoffs: { increase: ['More expressive state', 'Better long-range', 'More memory'], decrease: ['Faster', 'Less memory', 'May limit capacity'] }, research: 'Mamba uses N=16 as default. S4 originally used N=64 with HiPPO initialization.', recommendations: { efficient: '16', expressive: '32-64' } },
            { name: 'Conv Width (d_conv)', key: 'd_conv', category: 'architecture', type: 'slider', min: 2, max: 8, default: 4, step: 1, description: 'Width of the causal convolution before SSM. Provides local context before state space modeling. Standard is 4.', tip: '4 is standard from Mamba paper', compute: 'low', impact: 'low', tradeoffs: { increase: ['More local context', 'Slightly more compute'], decrease: ['Less local modeling', 'Faster'] }, research: 'Mamba ablations showed d_conv=4 works well. Values 2-4 are reasonable.', recommendations: { default: '4', minimal: '2', extended: '8' } },
            { name: 'Expansion Factor', key: 'expansion', category: 'architecture', type: 'slider', min: 1, max: 4, default: 2, step: 1, description: 'Inner dimension multiplier. Inner dim = expansion × d_model. Similar to FFN ratio in transformers but for SSM block.', tip: 'Inner dim = expansion × model_dim; 2 is standard', compute: 'high', impact: 'medium', tradeoffs: { increase: ['More capacity in SSM block', 'More parameters'], decrease: ['Faster', 'Fewer parameters'] }, research: 'Mamba uses expansion=2. This is where most computation happens.', recommendations: { default: '2', efficient: '1', large: '2-4' } },
            { name: 'Layers', key: 'layers', category: 'architecture', type: 'slider', min: 12, max: 64, default: 24, step: 4, description: 'Number of Mamba blocks. Scales similarly to transformer layers. Mamba can go deeper since no attention memory bottleneck.', tip: 'Similar scaling to Transformers', compute: 'high', impact: 'high', tradeoffs: { increase: ['More capacity', 'Deeper representations'], decrease: ['Faster', 'Easier optimization'] }, research: 'Mamba-1.4B uses 48 layers, Mamba-370M uses 24 layers.', recommendations: { small: '24', medium: '48', large: '64+' } },
            { name: 'Discretization', key: 'discretization', category: 'architecture', type: 'dropdown', options: ['ZOH', 'Bilinear'], default: 'ZOH', description: 'How to convert continuous SSM to discrete. ZOH (Zero-Order Hold) is standard and efficient. Bilinear is more accurate but slower.', tip: 'Zero-order hold is standard and efficient', compute: 'none', impact: 'low', tradeoffs: { ZOH: ['Efficient', 'Standard', 'Good approximation'], Bilinear: ['More accurate', 'Slower', 'Research option'] }, research: 'Mamba uses ZOH. S4 explored bilinear but ZOH dominates.', recommendations: { default: 'ZOH' } },
            { name: 'Initialization', key: 'initialization', category: 'training', type: 'dropdown', options: ['S4D', 'S4D-Real', 'Random'], default: 'S4D', description: 'How to initialize the SSM matrices (A, B, C). S4D uses diagonal approximation of HiPPO for stable long-range modeling.', tip: 'S4D crucial for long-range; HiPPO-derived', compute: 'none', impact: 'high', tradeoffs: { S4D: ['Stable long-range', 'HiPPO-based', 'Standard'], 'S4D-Real': ['Real-valued', 'Simpler', 'Good alternative'], Random: ['Simple', 'May not work well', 'Research only'] }, research: 'Gu et al. showed HiPPO initialization is crucial for long-range. S4D is diagonal approximation.', recommendations: { default: 'S4D', alternative: 'S4D-Real' } },
            { name: 'Activation', key: 'activation', category: 'architecture', type: 'dropdown', options: ['SiLU', 'GELU'], default: 'SiLU', description: 'Activation function in the Mamba block. SiLU (Swish) is standard for Mamba, consistent with modern LLM practice.', tip: 'SiLU standard for Mamba; consistent with LLMs', compute: 'low', impact: 'low', tradeoffs: { SiLU: ['Standard for Mamba', 'Smooth'], GELU: ['Alternative', 'Similar performance'] }, research: 'Mamba uses SiLU throughout, following LLaMA conventions.', recommendations: { default: 'SiLU' } },
            { name: 'Normalization', key: 'normalization', category: 'training', type: 'dropdown', options: ['RMSNorm', 'LayerNorm'], default: 'RMSNorm', description: 'Normalization before each Mamba block. RMSNorm is faster and standard for modern architectures.', tip: 'RMSNorm before each block; LLaMA-style', compute: 'low', impact: 'low', tradeoffs: { RMSNorm: ['Faster', 'Standard', 'No mean computation'], LayerNorm: ['Original', 'Slightly more compute'] }, research: 'Mamba uses RMSNorm following LLaMA. Applied before each block (Pre-Norm style).', recommendations: { default: 'RMSNorm' } },
            { name: 'Bidirectional', key: 'bidirectional', category: 'architecture', type: 'toggle', default: false, description: 'Process sequence in both directions. Useful for non-causal tasks like classification. Doubles computation.', tip: 'Enable for non-causal tasks; doubles compute', compute: 'high', impact: 'medium', tradeoffs: { enabled: ['Full context', 'Better for classification', '2x compute'], disabled: ['Causal', 'Generative ready', 'Faster'] }, research: 'Vision Mamba (Vim) uses bidirectional for image tasks.', recommendations: { generation: 'Disable', classification: 'Enable', vision: 'Enable' } }
        ]
    },
    {
        id: 'vits',
        name: 'Vision Transformers',
        icon: '🖼️',
        color: '#a855f7',
        description: 'Transformers adapted for image understanding',
        fullDescription: 'ViTs treat images as sequences of patches, applying transformer attention to learn global relationships. They require more data than CNNs but scale better and achieve superior results with sufficient training.',
        whenToUse: 'Image classification, when you have lots of data',
        innovation: 'Treats image patches as tokens for attention',
        typicalSize: '10M - 600M parameters',
        yearIntroduced: 2020,
        keyPapers: [
            { title: 'An Image is Worth 16x16 Words (ViT)', authors: 'Dosovitskiy et al.', year: 2020, arxiv: '2010.11929' },
            { title: 'DeiT: Training Data-Efficient Image Transformers', authors: 'Touvron et al.', year: 2021, arxiv: '2012.12877' },
            { title: 'Swin Transformer', authors: 'Liu et al.', year: 2021, arxiv: '2103.14030' }
        ],
        scalingGuidelines: {
            small: { params: '<30M', note: 'ViT-S, needs ImageNet-21k or strong augmentation' },
            medium: { params: '30-100M', note: 'ViT-B, standard for most tasks' },
            large: { params: '>100M', note: 'ViT-L/H, benefits from JFT-scale data' }
        },
        parameters: [
            { name: 'Patch Size', key: 'patch_size', category: 'architecture', type: 'dropdown', options: ['4', '8', '14', '16', '32'], default: '16', description: 'Size of image patches. Smaller patches = more tokens = higher compute but finer detail. 16x16 is standard.', tip: 'Smaller = more tokens, higher compute, finer detail', compute: 'varies', impact: 'high', tradeoffs: { smaller: ['Finer detail', 'Many more tokens', 'O(n²) attention cost'], larger: ['Fewer tokens', 'Faster', 'May lose detail'] }, research: 'ViT-16 means 16x16 patches. Smaller patches (8, 4) need more compute but often help.', recommendations: { standard: '16', high_res: '14', efficient: '32', detailed: '8' } },
            { name: 'Image Size', key: 'image_size', category: 'architecture', type: 'slider', min: 224, max: 512, default: 224, step: 32, description: 'Input image resolution. Higher resolution captures more detail but quadratically increases token count.', tip: '224 standard; 384/512 for fine-tuning boost', compute: 'high', impact: 'high', tradeoffs: { increase: ['More detail', 'Quadratic token increase', 'Better accuracy'], decrease: ['Much faster', 'May lose fine details'] }, research: 'Training at 224, fine-tuning at 384 is common. ViT-H uses 518 for SOTA.', recommendations: { training: '224', finetune: '384', maximum: '512-518' } },
            { name: 'Embed Dim', key: 'embed_dim', category: 'architecture', type: 'slider', min: 192, max: 1024, default: 768, step: 64, description: 'Token embedding dimension. Defines model width. Must be divisible by number of heads.', tip: 'ViT-B: 768, ViT-L: 1024, ViT-H: 1280', compute: 'high', impact: 'high', tradeoffs: { increase: ['Richer representations', 'More capacity'], decrease: ['Faster', 'Less memory'] }, research: 'ViT-B/16 uses 768, ViT-L uses 1024, ViT-H uses 1280.', recommendations: { tiny: '192', small: '384', base: '768', large: '1024' } },
            { name: 'Depth', key: 'depth', category: 'architecture', type: 'slider', min: 12, max: 32, default: 12, step: 1, description: 'Number of transformer blocks. ViT scales well with depth given sufficient data.', tip: 'ViT-B: 12, ViT-L: 24, ViT-H: 32', compute: 'high', impact: 'high', tradeoffs: { increase: ['Deeper features', 'Better with more data'], decrease: ['Faster', 'Easier optimization'] }, research: 'ViT-B uses 12 layers, ViT-L uses 24, ViT-H uses 32.', recommendations: { base: '12', large: '24', huge: '32' } },
            { name: 'Heads', key: 'heads', category: 'attention', type: 'slider', min: 3, max: 16, default: 12, step: 1, description: 'Number of attention heads. embed_dim must be divisible by heads. 64 per head is typical.', tip: 'embed_dim / 64 is common; ViT-B uses 12', compute: 'medium', impact: 'medium', tradeoffs: { increase: ['More attention patterns'], decrease: ['Larger head dimension'] }, research: 'ViT-B uses 12 heads with head_dim=64.', recommendations: { base: '12', large: '16' } },
            { name: 'MLP Ratio', key: 'mlp_ratio', category: 'architecture', type: 'slider', min: 2, max: 6, default: 4, step: 1, description: 'FFN hidden dimension as multiple of embed_dim. Standard is 4x, where most parameters reside.', tip: 'FFN hidden = mlp_ratio × embed_dim; 4 is standard', compute: 'high', impact: 'medium', tradeoffs: { increase: ['More FFN capacity', 'More parameters'], decrease: ['Smaller model', 'Faster'] }, research: 'All ViT variants use mlp_ratio=4 following transformer convention.', recommendations: { default: '4' } },
            { name: 'Conv Stem', key: 'conv_stem', category: 'architecture', type: 'toggle', default: false, description: 'Replace patch embedding with convolutional stem. Improves training stability and low-data performance.', tip: 'Improves stability and small-data performance', compute: 'low', impact: 'medium', tradeoffs: { enabled: ['Better optimization', 'Helps small data', 'Hybrid approach'], disabled: ['Pure ViT', 'Standard'] }, research: 'Xiao et al. (2021) showed conv stems improve ViT optimization. DeiT3 uses this.', recommendations: { limited_data: 'Enable', large_data: 'Optional' } },
            { name: 'Class Token', key: 'class_token', category: 'architecture', type: 'dropdown', options: ['CLS Token', 'GAP', 'Both'], default: 'CLS Token', description: 'How to get image representation. CLS token is learned. GAP (Global Average Pooling) averages all patch tokens.', tip: 'GAP often simpler; CLS is original design', compute: 'none', impact: 'low', tradeoffs: { 'CLS Token': ['Original design', 'Learned aggregation'], GAP: ['Simpler', 'No extra token', 'Often equally good'], Both: ['Ensemble', 'Slight boost'] }, research: 'ViT uses CLS token following BERT. Many find GAP equally effective.', recommendations: { default: 'CLS Token', alternative: 'GAP' } },
            { name: 'Pos Encoding', key: 'pos_encoding', category: 'architecture', type: 'dropdown', options: ['Learned', 'Sinusoidal', '2D', 'RoPE'], default: 'Learned', description: 'Position encoding type. Learned is standard. 2D encodes row/column separately. RoPE enables resolution flexibility.', tip: '2D for variable resolutions; RoPE emerging', compute: 'none', impact: 'medium', tradeoffs: { Learned: ['Standard', 'Fixed resolution', 'Simple'], Sinusoidal: ['No parameters', 'Fixed'], '2D': ['Row/column separate', 'Resolution flexible'], RoPE: ['Relative positions', 'Best flexibility'] }, research: 'ViT uses learned 1D positions. 2D and RoPE help with variable resolutions.', recommendations: { standard: 'Learned', flexible_resolution: '2D or RoPE' } },
            { name: 'Hierarchical', key: 'multi_stage', category: 'architecture', type: 'toggle', default: false, description: 'Multi-stage architecture like Swin with progressive downsampling. Better for dense prediction (detection, segmentation).', tip: 'Swin-style for detection/segmentation', compute: 'varies', impact: 'high', tradeoffs: { enabled: ['Multi-scale features', 'Dense prediction ready', 'Swin-style'], disabled: ['Simpler', 'Classification-focused', 'Original ViT'] }, research: 'Swin Transformer introduced hierarchical ViT with shifted windows for efficiency.', recommendations: { classification: 'Disable', detection: 'Enable (Swin)', segmentation: 'Enable (Swin)' } }
        ]
    },
    {
        id: 'vaes',
        name: 'Variational Autoencoders',
        icon: '🔄',
        color: '#22c55e',
        description: 'Probabilistic latent variable models',
        fullDescription: 'VAEs learn compressed latent representations by encoding inputs to distributions and decoding samples. The KL divergence regularizes the latent space, enabling interpolation and generation.',
        whenToUse: 'Representation learning, generation, anomaly detection',
        innovation: 'Learns compressed latent space with regularization',
        typicalSize: '100K - 10M parameters',
        yearIntroduced: 2013,
        keyPapers: [
            { title: 'Auto-Encoding Variational Bayes', authors: 'Kingma & Welling', year: 2013, arxiv: '1312.6114' },
            { title: 'beta-VAE: Learning Basic Visual Concepts', authors: 'Higgins et al.', year: 2017, link: 'ICLR 2017' },
            { title: 'NVAE: A Deep Hierarchical VAE', authors: 'Vahdat & Kautz', year: 2020, arxiv: '2007.03898' }
        ],
        scalingGuidelines: {
            small: { params: '<1M', note: 'Simple datasets, quick experiments' },
            medium: { params: '1-10M', note: 'Image generation, representation learning' },
            large: { params: '>10M', note: 'NVAE-style hierarchical for high quality' }
        },
        parameters: [
            { name: 'Latent Dim', key: 'latent_dim', category: 'architecture', type: 'slider', min: 2, max: 512, default: 128, step: 8, description: 'Dimension of the latent space. Higher = more capacity but harder to regularize. Sweet spot depends on data complexity.', tip: 'Higher = more capacity; balance with KL weight', compute: 'low', impact: 'high', tradeoffs: { increase: ['More capacity', 'Harder to regularize', 'Risk of posterior collapse'], decrease: ['Stronger compression', 'Better generalization', 'May lose detail'] }, research: 'MNIST works with 2-20 dims. Complex images need 128-512.', recommendations: { simple: '8-32', images: '64-128', complex: '256-512' } },
            { name: 'Encoder Layers', key: 'encoder_layers', category: 'architecture', type: 'slider', min: 2, max: 6, default: 4, step: 1, description: 'Depth of the encoder network. Usually symmetric with decoder.', tip: 'Symmetric with decoder is standard', compute: 'medium', impact: 'medium', tradeoffs: { increase: ['Better encoding', 'More capacity'], decrease: ['Faster', 'Simpler'] }, research: 'Standard VAEs use 3-5 layers. NVAE uses many more with hierarchy.', recommendations: { default: '4', deep: '5-6' } },
            { name: 'Decoder Layers', key: 'decoder_layers', category: 'architecture', type: 'slider', min: 2, max: 6, default: 4, step: 1, description: 'Depth of the decoder network. Often mirrors encoder architecture.', tip: 'Mirror encoder architecture', compute: 'medium', impact: 'medium', tradeoffs: { increase: ['Better reconstruction'], decrease: ['Faster generation'] }, research: 'Typically matches encoder depth.', recommendations: { default: '4' } },
            { name: 'Hidden Dim', key: 'hidden_dim', category: 'architecture', type: 'slider', min: 64, max: 1024, default: 256, step: 64, description: 'Hidden layer dimension in encoder/decoder. Often decreases toward bottleneck in encoder.', tip: 'Gradually decrease toward bottleneck', compute: 'medium', impact: 'medium', tradeoffs: { increase: ['More capacity'], decrease: ['Faster', 'Simpler'] }, research: 'Typical pyramid: 256→128→64→latent for encoder.', recommendations: { default: '256', small: '128', large: '512' } },
            { name: 'KL Weight (β)', key: 'kl_weight', category: 'training', type: 'slider', min: 0.001, max: 10, default: 1.0, step: 0.1, description: 'Weight on KL divergence term. β<1 prioritizes reconstruction (better quality). β>1 prioritizes disentanglement (beta-VAE).', tip: 'β<1 for quality; β>1 for disentanglement', compute: 'none', impact: 'high', tradeoffs: { increase: ['More disentangled', 'Worse reconstruction', 'beta-VAE regime'], decrease: ['Better reconstruction', 'Less regularized', 'May overfit'] }, research: 'Higgins et al. showed β>1 encourages disentanglement. β=1 is standard ELBO.', recommendations: { reconstruction: '0.1-0.5', balanced: '1.0', disentanglement: '4-10' } },
            { name: 'Annealing', key: 'annealing', category: 'training', type: 'toggle', default: true, description: 'Gradually increase KL weight during training. Prevents posterior collapse by letting reconstruction dominate initially.', tip: 'Gradually increase β to prevent collapse', compute: 'none', impact: 'medium', tradeoffs: { enabled: ['Prevents collapse', 'Better latent use', 'Curriculum approach'], disabled: ['Simpler', 'May cause collapse'] }, research: 'Bowman et al. (2016) showed KL annealing prevents posterior collapse.', recommendations: { default: 'Enable', stable_training: 'Optional' } },
            { name: 'Prior', key: 'prior', category: 'architecture', type: 'dropdown', options: ['Standard Normal', 'Learned', 'VampPrior', 'Flow'], default: 'Standard Normal', description: 'Prior distribution on latent space. Standard normal is simple. Learned priors can be more expressive.', tip: 'VampPrior for complex data; Normal is simple baseline', compute: 'varies', impact: 'medium', tradeoffs: { 'Standard Normal': ['Simple', 'Easy sampling', 'May not match data'], Learned: ['More flexible', 'Harder to sample'], VampPrior: ['Data-dependent', 'Expressive', 'More memory'], Flow: ['Most flexible', 'Normalizing flow prior', 'Complex'] }, research: 'Tomczak & Welling (2018) introduced VampPrior using pseudo-inputs.', recommendations: { default: 'Standard Normal', complex_data: 'VampPrior', maximum_flexibility: 'Flow' } },
            { name: 'Decoder Output', key: 'decoder_output', category: 'architecture', type: 'dropdown', options: ['Gaussian', 'Bernoulli', 'Categorical'], default: 'Gaussian', description: 'Output distribution type. Gaussian for continuous (images with MSE). Bernoulli for binary. Categorical for discrete.', tip: 'Gaussian for images; Bernoulli for binary data', compute: 'none', impact: 'medium', tradeoffs: { Gaussian: ['Continuous output', 'MSE loss', 'Standard for images'], Bernoulli: ['Binary output', 'BCE loss', 'MNIST-style'], Categorical: ['Discrete output', 'Cross-entropy'] }, research: 'Standard practice: Gaussian for normalized images, Bernoulli for binarized.', recommendations: { continuous: 'Gaussian', binary: 'Bernoulli', discrete: 'Categorical' } },
            { name: 'Free Bits', key: 'free_bits', category: 'training', type: 'slider', min: 0, max: 2, default: 0, step: 0.1, description: 'Minimum KL per dimension before it contributes to loss. Prevents posterior collapse by ensuring latent dimensions are used.', tip: 'Prevents collapse; 0.1-0.5 typical', compute: 'none', impact: 'medium', tradeoffs: { increase: ['Forces latent use', 'Prevents collapse', 'May hurt optimization'], decrease: ['Standard ELBO', 'May collapse'] }, research: 'Kingma et al. (2016) introduced free bits for sequence VAEs.', recommendations: { default: '0', anti_collapse: '0.1-0.5' } },
            { name: 'Architecture', key: 'architecture', category: 'architecture', type: 'dropdown', options: ['MLP', 'CNN', 'ResNet', 'Hierarchical'], default: 'CNN', description: 'Encoder/decoder architecture type. CNN for images. MLP for tabular. Hierarchical (NVAE) for high-quality generation.', tip: 'CNN for images; MLP for tabular; Hierarchical for quality', compute: 'varies', impact: 'high', tradeoffs: { MLP: ['Simple', 'Tabular data', 'Fast'], CNN: ['Spatial structure', 'Images', 'Standard'], ResNet: ['Deep', 'Skip connections', 'Better gradients'], Hierarchical: ['NVAE-style', 'Best quality', 'Complex'] }, research: 'NVAE showed hierarchical VAEs with residual cells achieve image quality.', recommendations: { tabular: 'MLP', images: 'CNN or ResNet', high_quality: 'Hierarchical' } }
        ]
    },
    {
        id: 'gans',
        name: 'GANs',
        icon: '⚔️',
        color: '#f97316',
        description: 'Adversarial training for generation',
        fullDescription: 'GANs train a generator and discriminator adversarially. The generator learns to fool the discriminator, while the discriminator learns to distinguish real from fake. This produces sharp, realistic outputs.',
        whenToUse: 'Image generation when you need sharp outputs',
        innovation: 'Generator and discriminator compete to improve',
        typicalSize: '1M - 100M parameters',
        yearIntroduced: 2014,
        keyPapers: [
            { title: 'Generative Adversarial Networks', authors: 'Goodfellow et al.', year: 2014, arxiv: '1406.2661' },
            { title: 'Progressive Growing of GANs', authors: 'Karras et al.', year: 2018, arxiv: '1710.10196' },
            { title: 'A Style-Based Generator Architecture (StyleGAN)', authors: 'Karras et al.', year: 2019, arxiv: '1812.04948' }
        ],
        scalingGuidelines: {
            small: { params: '<10M', note: 'Simple datasets, 64x64 resolution' },
            medium: { params: '10-50M', note: '256x256 resolution, good quality' },
            large: { params: '>50M', note: 'StyleGAN-scale, 1024x1024 resolution' }
        },
        parameters: [
            { name: 'Latent Dim (z)', key: 'latent_dim', category: 'architecture', type: 'slider', min: 64, max: 512, default: 128, step: 32, description: 'Dimension of the noise vector input to generator. Larger allows more variation but harder to cover.', tip: '128-512 typical; StyleGAN uses 512', compute: 'low', impact: 'medium', tradeoffs: { increase: ['More variation', 'Harder to train'], decrease: ['Easier to cover', 'Less diversity'] }, research: 'DCGAN used 100, StyleGAN uses 512. Most use 128-256.', recommendations: { default: '128-256', stylegan: '512' } },
            { name: 'Generator Layers', key: 'g_layers', category: 'architecture', type: 'slider', min: 4, max: 8, default: 5, step: 1, description: 'Number of upsampling layers in generator. Each doubles resolution. 5 layers: 4→8→16→32→64→128.', tip: 'Progressive growing helps high-resolution', compute: 'high', impact: 'high', tradeoffs: { increase: ['Higher resolution', 'More capacity'], decrease: ['Faster training', 'Lower resolution'] }, research: 'ProGAN showed progressive training helps. Each layer 2x resolution.', recommendations: { '64x64': '4', '256x256': '6', '1024x1024': '8' } },
            { name: 'Discriminator Layers', key: 'd_layers', category: 'architecture', type: 'slider', min: 4, max: 8, default: 5, step: 1, description: 'Number of downsampling layers in discriminator. Usually mirrors generator architecture.', tip: 'Match generator depth', compute: 'high', impact: 'high', tradeoffs: { increase: ['Stronger discriminator', 'Better gradients'], decrease: ['May be too weak'] }, research: 'Typically mirrors generator. Balance is crucial.', recommendations: { default: 'Match generator' } },
            { name: 'Base Channels', key: 'base_channels', category: 'architecture', type: 'slider', min: 32, max: 128, default: 64, step: 16, description: 'Channel count at finest resolution. Doubles at each coarser level (64→128→256→512...).', tip: 'Doubles at each resolution level', compute: 'high', impact: 'medium', tradeoffs: { increase: ['More capacity', 'Higher compute'], decrease: ['Faster', 'May underfit'] }, research: 'DCGAN used 64 base. StyleGAN uses 512 max channels with different scaling.', recommendations: { default: '64', high_quality: '128' } },
            { name: 'Normalization (G)', key: 'g_norm', category: 'training', type: 'dropdown', options: ['BatchNorm', 'InstanceNorm', 'None', 'AdaIN'], default: 'BatchNorm', description: 'Generator normalization. BatchNorm is standard. AdaIN enables StyleGAN-style modulation.', tip: 'AdaIN for StyleGAN; BatchNorm for standard', compute: 'low', impact: 'medium', tradeoffs: { BatchNorm: ['Standard', 'Batch-dependent'], InstanceNorm: ['Per-image', 'Style transfer'], None: ['Simplest'], AdaIN: ['Style modulation', 'StyleGAN key'] }, research: 'StyleGAN uses AdaIN for style injection. Standard GANs use BatchNorm.', recommendations: { standard: 'BatchNorm', stylegan: 'AdaIN' } },
            { name: 'Normalization (D)', key: 'd_norm', category: 'training', type: 'dropdown', options: ['None', 'SpectralNorm', 'LayerNorm'], default: 'SpectralNorm', description: 'Discriminator normalization. SpectralNorm constrains Lipschitz constant, stabilizing training. Do NOT use BatchNorm in D.', tip: 'SpectralNorm stabilizes D; never use BatchNorm in D', compute: 'low', impact: 'high', tradeoffs: { None: ['Simplest', 'May be unstable'], SpectralNorm: ['Lipschitz constraint', 'Very stable', 'Standard'], LayerNorm: ['Alternative', 'Per-sample'] }, research: 'Miyato et al. (2018) showed SpectralNorm dramatically stabilizes GANs.', recommendations: { default: 'SpectralNorm' } },
            { name: 'Loss', key: 'loss', category: 'training', type: 'dropdown', options: ['Vanilla', 'WGAN', 'WGAN-GP', 'Hinge', 'Least Squares'], default: 'Hinge', description: 'Adversarial loss function. Hinge is simple and stable. WGAN-GP uses gradient penalty for Lipschitz constraint.', tip: 'Hinge loss stable and simple; WGAN-GP reliable', compute: 'low', impact: 'high', tradeoffs: { Vanilla: ['Original', 'Can be unstable', 'Mode collapse risk'], WGAN: ['Wasserstein distance', 'Needs weight clipping'], 'WGAN-GP': ['Gradient penalty', 'Very stable', 'Slower'], Hinge: ['Simple', 'Stable', 'BigGAN default'], 'Least Squares': ['Smooth gradients', 'Less mode collapse'] }, research: 'BigGAN uses hinge loss. WGAN-GP (Gulrajani et al., 2017) is very stable.', recommendations: { default: 'Hinge', maximum_stability: 'WGAN-GP' } },
            { name: 'Learning Rate (G)', key: 'g_lr', category: 'training', type: 'slider', min: 0.0001, max: 0.001, default: 0.0002, step: 0.0001, description: 'Generator learning rate. Often slightly lower than D to prevent mode collapse.', tip: 'Often lower than D; TTUR uses different rates', compute: 'none', impact: 'high', tradeoffs: { increase: ['Faster G updates', 'Risk mode collapse'], decrease: ['Slower but stable', 'D stays ahead'] }, research: 'TTUR (Heusel et al., 2017) suggests different LRs for G and D.', recommendations: { default: '0.0002', conservative: '0.0001' } },
            { name: 'Learning Rate (D)', key: 'd_lr', category: 'training', type: 'slider', min: 0.0001, max: 0.001, default: 0.0002, step: 0.0001, description: 'Discriminator learning rate. Can be higher than G to maintain strong gradients.', tip: 'Can be higher than G for stronger gradients', compute: 'none', impact: 'high', tradeoffs: { increase: ['Stronger D', 'Better gradients for G'], decrease: ['Weaker D', 'G may overpower'] }, research: 'TTUR often uses higher D learning rate (e.g., 4x).', recommendations: { default: '0.0002', TTUR: '0.0004' } },
            { name: 'D Updates per G', key: 'd_updates', category: 'training', type: 'slider', min: 1, max: 5, default: 1, step: 1, description: 'Discriminator updates per generator update. WGAN originally used 5. Most modern GANs use 1.', tip: 'WGAN: 5; most modern GANs: 1', compute: 'varies', impact: 'medium', tradeoffs: { increase: ['Stronger D', 'Better gradients', 'Slower'], decrease: ['Balanced', 'Faster', 'Standard'] }, research: 'Original WGAN used 5. With SpectralNorm and hinge loss, 1 is sufficient.', recommendations: { WGAN: '5', modern: '1' } }
        ]
    }
];
