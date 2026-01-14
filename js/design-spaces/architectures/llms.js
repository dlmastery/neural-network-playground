/**
 * Large Language Models Architecture Configuration
 * Scaled transformer models for text generation
 */

export const llms = {
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
        },
        {
            name: 'Mixed Precision',
            key: 'mixed_precision',
            category: 'efficiency',
            type: 'dropdown',
            options: ['FP32', 'FP16', 'BF16', 'FP8'],
            default: 'BF16',
            description: 'Floating-point precision for training. BF16 has same range as FP32 but less precision. FP8 is emerging for inference.',
            tip: 'BF16 standard for training; FP8 for inference',
            compute: 'varies',
            impact: 'high',
            tradeoffs: {
                FP32: ['Maximum precision', 'Slowest', '2x memory', 'Baseline'],
                FP16: ['2x faster', 'Half memory', 'Needs loss scaling', 'Overflow risk'],
                BF16: ['2x faster', 'No overflow', 'Same range as FP32', 'Industry standard'],
                FP8: ['4x faster', 'Quarter memory', 'Needs calibration', 'Inference only']
            },
            research: 'Micikevicius et al. (2017) introduced mixed precision. BF16 (Google, 2019) avoids overflow issues. FP8 (NVIDIA, 2022) emerging.',
            recommendations: {
                training: 'BF16 (standard)',
                inference: 'FP16 or FP8',
                debugging: 'FP32'
            }
        },
        {
            name: 'Gradient Accumulation',
            key: 'gradient_accumulation',
            category: 'training',
            type: 'slider',
            min: 1,
            max: 128,
            default: 8,
            step: 1,
            description: 'Number of micro-batches to accumulate before optimizer step. Enables large effective batch sizes on limited memory.',
            tip: 'Effective batch = micro_batch × accumulation × GPUs',
            compute: 'none',
            impact: 'medium',
            tradeoffs: {
                increase: ['Larger effective batch', 'Fits in memory', 'Slower step frequency'],
                decrease: ['More frequent updates', 'Less memory efficient', 'May need smaller LR']
            },
            research: 'Standard technique for distributed training. LLaMA used batch size of 4M tokens via accumulation across many GPUs.',
            recommendations: {
                single_gpu: '8-32',
                multi_gpu: '4-16 per GPU',
                large_batch: 'Target 1-4M tokens total'
            }
        },
        {
            name: 'Batch Size (tokens)',
            key: 'batch_size',
            category: 'training',
            type: 'dropdown',
            options: ['64K', '256K', '512K', '1M', '2M', '4M'],
            default: '1M',
            description: 'Total tokens per optimizer step across all GPUs. Larger batches enable higher learning rates and smoother training.',
            tip: '1-4M tokens standard for LLMs; scale with model size',
            compute: 'varies',
            impact: 'high',
            tradeoffs: {
                increase: ['Smoother gradients', 'Higher LR possible', 'Better parallelization', 'More memory'],
                decrease: ['Noisier gradients', 'Lower LR needed', 'Less parallelization', 'May generalize better']
            },
            research: 'Scaling laws show critical batch size increases with model size. GPT-3 used 3.2M tokens, LLaMA used 4M tokens.',
            recommendations: {
                small_model: '256K-512K tokens',
                medium_model: '1M-2M tokens',
                large_model: '2M-4M tokens'
            }
        },
        {
            name: 'LR Schedule',
            key: 'lr_schedule',
            category: 'training',
            type: 'dropdown',
            options: ['Constant', 'Linear Decay', 'Cosine', 'Cosine with Restarts', 'WSD'],
            default: 'Cosine',
            description: 'Learning rate decay schedule after warmup. Cosine decay is standard for LLMs. WSD (Warmup-Stable-Decay) is emerging.',
            tip: 'Cosine standard; WSD for continued pretraining',
            compute: 'none',
            impact: 'medium',
            tradeoffs: {
                Constant: ['Simple', 'Good for fine-tuning', 'May overfit late'],
                'Linear Decay': ['Smooth decay', 'Predictable', 'Less common'],
                Cosine: ['Smooth to near-zero', 'Most studied', 'Industry standard'],
                'Cosine with Restarts': ['Multiple cycles', 'Good for long training', 'Complex'],
                WSD: ['Stable plateau phase', 'Good for continued training', 'MiniCPM approach']
            },
            research: 'Loshchilov & Hutter (2016) showed cosine annealing improves results. WSD from MiniCPM enables better continued pretraining.',
            recommendations: {
                pretraining: 'Cosine',
                fine_tuning: 'Constant or Linear Decay',
                continued_pretraining: 'WSD'
            }
        }
    ]
};
