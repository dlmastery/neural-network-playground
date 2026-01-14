/**
 * SSMs / Mamba Architecture Configuration
 * State space models with selective mechanisms
 */

export const ssms = {
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
};
