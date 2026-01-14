/**
 * Graph Neural Networks Architecture Configuration
 * Neural networks for graph-structured data
 */

export const gnns = {
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
};
