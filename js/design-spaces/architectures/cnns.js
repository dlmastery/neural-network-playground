/**
 * Convolutional Neural Networks Architecture Configuration
 * Spatial feature extractors for images
 */

export const cnns = {
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
        { name: 'Depthwise Separable', key: 'depthwise', category: 'efficiency', type: 'toggle', default: false, description: 'Split convolution into depthwise (per-channel) and pointwise (1x1) operations. Reduces computation by ~8-9x.', tip: 'Essential for mobile efficiency (MobileNet)', compute: 'low', impact: 'medium', tradeoffs: { enabled: ['8-9x fewer FLOPs', 'Mobile-friendly', 'Slight accuracy drop'], disabled: ['Full convolutions', 'Maximum accuracy', 'More compute'] }, research: 'Howard et al. (2017) MobileNet showed depthwise separable enables mobile deployment.', recommendations: { server: 'Disable', mobile: 'Enable', edge: 'Enable' } },
        {
            name: 'Stochastic Depth',
            key: 'stochastic_depth',
            category: 'regularization',
            type: 'slider',
            min: 0,
            max: 0.5,
            default: 0.1,
            step: 0.05,
            description: 'Randomly drops entire residual blocks during training (survival prob = 1 - rate). Provides regularization and reduces training time. Rate increases linearly from 0 at input to max at output.',
            tip: '0.1-0.2 typical; higher for larger models',
            compute: 'none',
            impact: 'medium',
            tradeoffs: {
                increase: ['Stronger regularization', 'Faster training', 'May hurt small models'],
                decrease: ['Less regularization', 'Full depth always', 'Standard behavior']
            },
            research: 'Huang et al. (2016) showed stochastic depth improves ResNet training. Now standard in EfficientNet, ConvNeXt.',
            recommendations: {
                small_model: '0.05-0.1',
                medium_model: '0.1-0.2',
                large_model: '0.2-0.5'
            }
        },
        {
            name: 'Data Augmentation',
            key: 'augmentation',
            category: 'training',
            type: 'dropdown',
            options: ['Basic', 'AutoAugment', 'RandAugment', 'TrivialAugment'],
            default: 'RandAugment',
            description: 'Augmentation strategy during training. RandAugment applies N random transforms at magnitude M. AutoAugment is learned but expensive to search.',
            tip: 'RandAugment(N=2, M=9) is simple and effective',
            compute: 'low',
            impact: 'high',
            tradeoffs: {
                Basic: ['Flips, crops, color jitter', 'Fast', 'Limited diversity'],
                AutoAugment: ['Learned policy', 'Dataset-specific', 'Search expensive'],
                RandAugment: ['Simple random', 'N and M params only', 'State-of-art'],
                TrivialAugment: ['Single transform', 'Simpler than RandAug', 'Competitive']
            },
            research: 'Cubuk et al. (2020) showed RandAugment matches AutoAugment with much simpler search. N=2, M=9 is robust default.',
            recommendations: {
                simple: 'Basic',
                standard: 'RandAugment (N=2, M=9)',
                maximum: 'AutoAugment or TrivialAugment'
            }
        },
        {
            name: 'Mixup/CutMix',
            key: 'mixup',
            category: 'regularization',
            type: 'dropdown',
            options: ['None', 'Mixup', 'CutMix', 'Both'],
            default: 'Both',
            description: 'Advanced data augmentation that mixes training samples. Mixup blends images and labels. CutMix patches one image onto another. Using both alternately gives best results.',
            tip: 'CutMix better for localization; both for classification',
            compute: 'low',
            impact: 'high',
            tradeoffs: {
                None: ['Standard training', 'No mixing'],
                Mixup: ['Global blending', 'Smooth decision boundaries', 'α=0.2-0.4'],
                CutMix: ['Patch-based', 'Better localization', 'α=1.0'],
                Both: ['Alternate randomly', 'Best results', 'Typical choice']
            },
            research: 'Zhang et al. (2018) introduced Mixup. Yun et al. (2019) showed CutMix improves localization. Combining gives best ImageNet results.',
            recommendations: {
                small_data: 'Both',
                large_data: 'Both or CutMix',
                localization: 'CutMix'
            }
        },
        {
            name: 'Label Smoothing',
            key: 'label_smoothing',
            category: 'regularization',
            type: 'slider',
            min: 0,
            max: 0.3,
            default: 0.1,
            step: 0.05,
            description: 'Softens one-hot labels by distributing ε probability to non-target classes. Prevents overconfidence and improves calibration.',
            tip: '0.1 standard for ImageNet; 0 for distillation',
            compute: 'none',
            impact: 'medium',
            tradeoffs: {
                increase: ['Better calibration', 'Less overconfident', 'May hurt accuracy'],
                decrease: ['Hard labels', 'Standard training', 'More confident']
            },
            research: 'Szegedy et al. (2016) introduced label smoothing for Inception. Müller et al. (2019) analyzed why it works.',
            recommendations: {
                default: '0.1',
                small_dataset: '0.0-0.05',
                distillation: '0.0'
            }
        },
        {
            name: 'Weight Decay',
            key: 'weight_decay',
            category: 'training',
            type: 'slider',
            min: 0,
            max: 0.1,
            default: 0.0001,
            step: 0.00005,
            description: 'L2 regularization coefficient. Modern optimizers (AdamW) decouple weight decay from gradient updates for better results.',
            tip: 'AdamW: 0.01-0.1; SGD: 1e-4 to 5e-4',
            compute: 'none',
            impact: 'medium',
            tradeoffs: {
                increase: ['Stronger regularization', 'Smaller weights', 'May underfit'],
                decrease: ['Less regularization', 'Larger weights', 'May overfit']
            },
            research: 'Loshchilov & Hutter (2019) showed decoupled weight decay (AdamW) outperforms L2 regularization in Adam.',
            recommendations: {
                SGD: '1e-4 to 5e-4',
                AdamW: '0.01 to 0.1',
                fine_tuning: '0.01'
            }
        },
        {
            name: 'Learning Rate',
            key: 'learning_rate',
            category: 'training',
            type: 'slider',
            min: 0.0001,
            max: 0.5,
            default: 0.1,
            step: 0.001,
            description: 'Initial learning rate before scheduling. SGD typically uses higher LR (0.1) while Adam uses lower (0.001).',
            tip: 'SGD: 0.1; AdamW: 1e-3 to 4e-3',
            compute: 'none',
            impact: 'high',
            tradeoffs: {
                increase: ['Faster training', 'May diverge', 'Larger steps'],
                decrease: ['More stable', 'Slower convergence', 'May underfit']
            },
            research: 'Linear scaling rule: LR ∝ batch_size. Smith (2018) showed super-convergence with cyclical LR.',
            recommendations: {
                SGD: '0.1 (ImageNet)',
                AdamW: '1e-3 to 4e-3',
                fine_tuning: '1e-4 to 1e-5'
            }
        },
        {
            name: 'LR Schedule',
            key: 'lr_schedule',
            category: 'training',
            type: 'dropdown',
            options: ['Step', 'Cosine', 'OneCycle', 'Warmup + Cosine'],
            default: 'Warmup + Cosine',
            description: 'Learning rate decay schedule. Cosine annealing is standard for modern CNNs. OneCycle enables faster training.',
            tip: 'Cosine standard; OneCycle for speed',
            compute: 'none',
            impact: 'medium',
            tradeoffs: {
                Step: ['Classic ResNet', 'Discrete drops', 'Requires tuning'],
                Cosine: ['Smooth decay', 'No hyperparams', 'Standard'],
                OneCycle: ['Fast convergence', 'Super-convergence', 'Complex'],
                'Warmup + Cosine': ['Linear warmup then cosine', 'Most robust', 'Modern default']
            },
            research: 'Loshchilov & Hutter (2016) showed cosine annealing improves results. Smith (2018) introduced OneCycle.',
            recommendations: {
                classic: 'Step (30, 60, 90 epochs)',
                modern: 'Warmup + Cosine',
                fast: 'OneCycle'
            }
        }
    ]
};
