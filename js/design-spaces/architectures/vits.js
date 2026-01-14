/**
 * Vision Transformers Architecture Configuration
 * Transformers adapted for image understanding
 */

export const vits = {
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
        { name: 'Hierarchical', key: 'multi_stage', category: 'architecture', type: 'toggle', default: false, description: 'Multi-stage architecture like Swin with progressive downsampling. Better for dense prediction (detection, segmentation).', tip: 'Swin-style for detection/segmentation', compute: 'varies', impact: 'high', tradeoffs: { enabled: ['Multi-scale features', 'Dense prediction ready', 'Swin-style'], disabled: ['Simpler', 'Classification-focused', 'Original ViT'] }, research: 'Swin Transformer introduced hierarchical ViT with shifted windows for efficiency.', recommendations: { classification: 'Disable', detection: 'Enable (Swin)', segmentation: 'Enable (Swin)' } },
        {
            name: 'Layer Scale',
            key: 'layer_scale',
            category: 'training',
            type: 'slider',
            min: 0,
            max: 0.1,
            default: 0.0001,
            step: 0.0001,
            description: 'Initial value for learnable per-channel scaling after each residual block. Helps train deeper ViTs by starting with near-identity blocks. 0 disables.',
            tip: 'Essential for training deep ViTs (24+ layers)',
            compute: 'none',
            impact: 'high',
            tradeoffs: {
                increase: ['Slower adaptation', 'More stable start'],
                decrease: ['Faster adaptation', 'May destabilize deep nets']
            },
            research: 'Touvron et al. (2021) DeiT showed layer scale (1e-4 to 1e-6) is critical for training deep ViTs without divergence.',
            recommendations: {
                shallow: '0 (not needed)',
                deep: '1e-4 (DeiT default)',
                very_deep: '1e-6'
            }
        },
        {
            name: 'Drop Path',
            key: 'drop_path',
            category: 'regularization',
            type: 'slider',
            min: 0,
            max: 0.5,
            default: 0.1,
            step: 0.05,
            description: 'Stochastic depth rate - probability of dropping paths in residual connections. Rate increases linearly with depth. Critical regularizer for ViTs.',
            tip: '0.1-0.2 for Base, 0.4+ for Large models',
            compute: 'none',
            impact: 'high',
            tradeoffs: {
                increase: ['Stronger regularization', 'Prevents overfitting', 'May hurt small data'],
                decrease: ['Less regularization', 'Full model capacity', 'May overfit']
            },
            research: 'DeiT uses 0.1 for ViT-B. Larger models need more (0.4-0.5). Rate increases linearly from 0 at layer 1.',
            recommendations: {
                ViT_S: '0.1',
                ViT_B: '0.1-0.2',
                ViT_L: '0.4-0.5'
            }
        },
        {
            name: 'Data Augmentation',
            key: 'augmentation',
            category: 'training',
            type: 'dropdown',
            options: ['Basic', 'RandAugment', 'AutoAugment', '3-Augment'],
            default: 'RandAugment',
            description: 'Augmentation strategy. ViTs need stronger augmentation than CNNs due to lack of inductive bias. 3-Augment is DeiT3 recipe.',
            tip: 'ViTs require strong augmentation; 3-Augment for DeiT3',
            compute: 'low',
            impact: 'high',
            tradeoffs: {
                Basic: ['Insufficient for ViT', 'May overfit', 'Not recommended'],
                RandAugment: ['Standard choice', 'N=2, M=9', 'DeiT default'],
                AutoAugment: ['Learned policy', 'Slightly better', 'More complex'],
                '3-Augment': ['DeiT3 recipe', 'Simpler than RandAug', 'Grayscale+Solarize+GaussianBlur']
            },
            research: 'Touvron et al. (2022) DeiT3 showed 3-Augment is simpler and matches RandAugment. ViTs need augmentation more than CNNs.',
            recommendations: {
                standard: 'RandAugment (N=2, M=9)',
                simple: '3-Augment',
                maximum: 'AutoAugment'
            }
        },
        {
            name: 'Mixup/CutMix',
            key: 'mixup',
            category: 'regularization',
            type: 'dropdown',
            options: ['None', 'Mixup', 'CutMix', 'Both'],
            default: 'Both',
            description: 'Sample mixing augmentation. Critical for ViT training. DeiT uses 0.8 mixup and 1.0 cutmix with 0.5 switching probability.',
            tip: 'Essential for ViT; stronger than for CNNs',
            compute: 'low',
            impact: 'high',
            tradeoffs: {
                None: ['Not recommended for ViT', 'Will overfit'],
                Mixup: ['Smooth boundaries', 'α=0.8 for ViT'],
                CutMix: ['Patch-based', 'α=1.0 for ViT'],
                Both: ['Best results', 'DeiT default', 'p=0.5 switch']
            },
            research: 'DeiT showed mixup/cutmix is critical for ViT when not using JFT-scale data.',
            recommendations: {
                always: 'Both (α_mixup=0.8, α_cutmix=1.0)'
            }
        },
        {
            name: 'Label Smoothing',
            key: 'label_smoothing',
            category: 'regularization',
            type: 'slider',
            min: 0,
            max: 0.2,
            default: 0.1,
            step: 0.02,
            description: 'Smooths hard labels by distributing probability. Standard 0.1 for ImageNet classification.',
            tip: '0.1 standard; helps calibration',
            compute: 'none',
            impact: 'medium',
            tradeoffs: {
                increase: ['Better calibration', 'May hurt accuracy'],
                decrease: ['Hard labels', 'More confident predictions']
            },
            research: 'DeiT uses 0.1 label smoothing consistently.',
            recommendations: {
                default: '0.1'
            }
        },
        {
            name: 'Weight Decay',
            key: 'weight_decay',
            category: 'training',
            type: 'slider',
            min: 0,
            max: 0.3,
            default: 0.05,
            step: 0.01,
            description: 'AdamW weight decay. ViTs use higher weight decay (0.05-0.3) than CNNs. Applied to all params except biases and LayerNorm.',
            tip: 'ViTs need high weight decay (0.05-0.3)',
            compute: 'none',
            impact: 'high',
            tradeoffs: {
                increase: ['Stronger regularization', 'Smaller weights', 'May underfit'],
                decrease: ['Less regularization', 'May overfit', 'Larger weights']
            },
            research: 'DeiT uses 0.05 for ViT-B, 0.3 for ViT-L. Much higher than CNN defaults.',
            recommendations: {
                ViT_S: '0.05',
                ViT_B: '0.05',
                ViT_L: '0.1-0.3',
                fine_tuning: '0.05'
            }
        },
        {
            name: 'Learning Rate',
            key: 'learning_rate',
            category: 'training',
            type: 'slider',
            min: 0.0001,
            max: 0.003,
            default: 0.001,
            step: 0.0001,
            description: 'Peak learning rate with AdamW. ViTs typically use 1e-3 to 3e-3 with warmup + cosine decay.',
            tip: '1e-3 typical; scale with batch size',
            compute: 'none',
            impact: 'high',
            tradeoffs: {
                increase: ['Faster learning', 'May diverge'],
                decrease: ['More stable', 'Slower convergence']
            },
            research: 'DeiT uses 1e-3 base LR with linear scaling for batch size. LR = base_lr × batch_size / 512.',
            recommendations: {
                default: '1e-3',
                large_batch: '3e-3',
                fine_tuning: '1e-5 to 1e-4'
            }
        },
        {
            name: 'Warmup Epochs',
            key: 'warmup_epochs',
            category: 'training',
            type: 'slider',
            min: 0,
            max: 50,
            default: 5,
            step: 1,
            description: 'Linear warmup epochs before cosine decay. Longer warmup helps stabilize ViT training.',
            tip: '5 epochs standard; more for larger models',
            compute: 'none',
            impact: 'medium',
            tradeoffs: {
                increase: ['More stable start', 'Longer warmup phase'],
                decrease: ['Faster training', 'May be unstable early']
            },
            research: 'DeiT uses 5 warmup epochs out of 300 total. Some use 10-20 for very large models.',
            recommendations: {
                standard: '5 epochs',
                large_model: '10-20 epochs',
                fine_tuning: '0-1 epochs'
            }
        },
        {
            name: 'EMA',
            key: 'ema',
            category: 'training',
            type: 'toggle',
            default: true,
            description: 'Exponential Moving Average of model weights. Use EMA model for evaluation. Provides smoother, more robust predictions.',
            tip: 'Always use for inference; 0.9999 decay typical',
            compute: 'low',
            impact: 'medium',
            tradeoffs: {
                enabled: ['Smoother weights', 'Better generalization', 'Small memory overhead'],
                disabled: ['Use final weights', 'Simpler', 'Slightly worse']
            },
            research: 'EMA with 0.9999 decay is standard for ViT training. ~0.5% improvement on ImageNet.',
            recommendations: {
                default: 'Enable with 0.9999 decay'
            }
        }
    ]
};
