/**
 * GANs Architecture Configuration
 * Adversarial training for generation
 */

export const gans = {
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
};
