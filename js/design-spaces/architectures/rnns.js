/**
 * RNNs & LSTMs Architecture Configuration
 * Recurrent networks for sequential data
 */

export const rnns = {
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
};
