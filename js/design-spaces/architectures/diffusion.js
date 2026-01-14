/**
 * Diffusion Models Architecture Configuration
 * Denoising-based generative models
 */

export const diffusion = {
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
        },
        {
            name: 'VAE',
            key: 'vae',
            category: 'architecture',
            type: 'dropdown',
            options: ['None (Pixel)', 'KL-VAE', 'VQ-VAE', 'SD-VAE', 'SDXL-VAE'],
            default: 'SD-VAE',
            description: 'Variational Autoencoder for latent diffusion. Compresses images to latent space (typically 8x), enabling faster training and inference with lower memory.',
            tip: 'Latent diffusion (with VAE) is 10-100x more efficient than pixel diffusion',
            compute: 'medium',
            impact: 'high',
            tradeoffs: {
                'None (Pixel)': ['Maximum fidelity', 'Very slow', 'High memory', 'Research only'],
                'KL-VAE': ['Continuous latent', 'Standard choice', 'Good reconstruction'],
                'VQ-VAE': ['Discrete tokens', 'Used by DALL-E', 'Codebook collapse risk'],
                'SD-VAE': ['8x compression', 'Good balance', 'Industry standard'],
                'SDXL-VAE': ['Improved fidelity', 'Better colors', 'SDXL default']
            },
            research: 'Rombach et al. (2022) showed latent diffusion achieves similar quality at fraction of compute. SD uses f=8 (64x64 latent for 512x512).',
            recommendations: {
                research: 'None (Pixel) for ablations',
                standard: 'SD-VAE (f=8)',
                high_fidelity: 'SDXL-VAE'
            }
        },
        {
            name: 'Prediction Type',
            key: 'prediction_type',
            category: 'training',
            type: 'dropdown',
            options: ['Epsilon', 'V-Prediction', 'Sample'],
            default: 'Epsilon',
            description: 'What the model predicts at each step. Epsilon predicts noise. V-prediction predicts a velocity vector. Sample directly predicts the clean image.',
            tip: 'V-prediction more stable at high guidance scales',
            compute: 'none',
            impact: 'medium',
            tradeoffs: {
                Epsilon: ['Original formulation', 'Most common', 'Works well at low CFG'],
                'V-Prediction': ['More stable high CFG', 'Better for video', 'SD 2.0+ option'],
                Sample: ['Direct prediction', 'Simpler interpretation', 'Less common']
            },
            research: 'Salimans & Ho (2022) showed v-prediction improves stability. SD 2.0 offers both epsilon and v-pred checkpoints.',
            recommendations: {
                standard: 'Epsilon',
                high_cfg: 'V-Prediction',
                video: 'V-Prediction'
            }
        },
        {
            name: 'EMA Decay',
            key: 'ema_decay',
            category: 'training',
            type: 'slider',
            min: 0.9,
            max: 0.9999,
            default: 0.9999,
            step: 0.0001,
            description: 'Exponential Moving Average decay for model weights. EMA weights are used for inference, providing smoother outputs than training weights.',
            tip: 'Always use EMA weights for inference; 0.9999 standard',
            compute: 'low',
            impact: 'medium',
            tradeoffs: {
                increase: ['Smoother weights', 'Better quality', 'Slower adaptation'],
                decrease: ['Faster adaptation', 'More noise', 'Tracks training closely']
            },
            research: 'Karras et al. (2022) analyzed EMA schedules. High decay (0.9999) is standard. Some use adaptive EMA.',
            recommendations: {
                standard: '0.9999',
                fast_adaptation: '0.999',
                very_smooth: '0.99999'
            }
        },
        {
            name: 'CFG Rescale',
            key: 'cfg_rescale',
            category: 'generation',
            type: 'slider',
            min: 0,
            max: 1,
            default: 0,
            step: 0.1,
            description: 'Rescales the CFG output to reduce artifacts at high guidance scales. Higher values reduce color oversaturation and burned highlights.',
            tip: '0.7 helps with overexposure at CFG > 10',
            compute: 'none',
            impact: 'medium',
            tradeoffs: {
                increase: ['Reduces oversaturation', 'Softer outputs', 'May reduce contrast'],
                decrease: ['Standard CFG behavior', 'Can oversaturate', 'More vibrant']
            },
            research: 'Lin et al. (2024) introduced CFG rescaling to address guidance artifacts. Implemented in modern UIs like ComfyUI.',
            recommendations: {
                low_cfg: '0 (not needed)',
                high_cfg: '0.7',
                very_high_cfg: '0.8-0.9'
            }
        },
        {
            name: 'LoRA Rank',
            key: 'lora_rank',
            category: 'efficiency',
            type: 'dropdown',
            options: ['None', '4', '8', '16', '32', '64', '128'],
            default: 'None',
            description: 'Low-Rank Adaptation rank for fine-tuning. LoRA adds small trainable matrices to frozen weights, enabling efficient personalization with <1% parameters.',
            tip: 'Rank 8-32 sufficient for most fine-tuning',
            compute: 'low',
            impact: 'medium',
            tradeoffs: {
                None: ['Full training or inference only'],
                '4': ['Minimal capacity', 'Very fast', 'Style transfer'],
                '8': ['Good for styles', 'Fast training', 'Most efficient'],
                '16': ['Standard choice', 'Good balance', 'Characters/concepts'],
                '32': ['Higher capacity', 'Complex subjects', 'Longer training'],
                '64': ['Near full fine-tune', 'Very specific', 'Slow'],
                '128': ['Maximum capacity', 'Near full model', 'Rarely needed']
            },
            research: 'Hu et al. (2021) introduced LoRA. Rank 8-16 captures most variation for SD models.',
            recommendations: {
                style: '4-8',
                character: '16-32',
                complex: '32-64'
            }
        },
        {
            name: 'SNR Weighting',
            key: 'snr_weighting',
            category: 'training',
            type: 'dropdown',
            options: ['None', 'Min-SNR', 'Truncated SNR', 'Soft Min-SNR'],
            default: 'Min-SNR',
            description: 'Signal-to-Noise Ratio weighting for loss function. Helps balance learning across timesteps by downweighting very noisy steps.',
            tip: 'Min-SNR γ=5 improves training stability',
            compute: 'none',
            impact: 'medium',
            tradeoffs: {
                None: ['Standard uniform weighting', 'May overtrain on easy steps'],
                'Min-SNR': ['Better balance', 'Faster convergence', 'γ=5 default'],
                'Truncated SNR': ['Hard cutoff', 'Ignores very noisy steps'],
                'Soft Min-SNR': ['Smooth weighting', 'Gentle transitions']
            },
            research: 'Hang et al. (2023) showed Min-SNR weighting improves FID. Implemented in diffusers library.',
            recommendations: {
                default: 'Min-SNR (γ=5)',
                aggressive: 'Truncated SNR',
                conservative: 'None'
            }
        },
        {
            name: 'Resolution',
            key: 'resolution',
            category: 'architecture',
            type: 'dropdown',
            options: ['256', '512', '768', '1024', '2048'],
            default: '512',
            description: 'Native training resolution. Higher resolutions require more compute but produce more detailed outputs. Can generate at other sizes with some quality loss.',
            tip: '512 for SD 1.5, 1024 for SDXL, can upscale with img2img',
            compute: 'high',
            impact: 'high',
            tradeoffs: {
                '256': ['Fast prototyping', 'Low quality', 'Research only'],
                '512': ['SD 1.5 native', 'Good balance', 'Industry standard'],
                '768': ['SD 2.0 native', 'Better detail', 'More memory'],
                '1024': ['SDXL native', 'High detail', '4x compute of 512'],
                '2048': ['Ultra high-res', 'Requires tiling/upscaling', 'Very expensive']
            },
            research: 'SDXL (2023) showed 1024 significantly improves hands/faces. Higher res needs careful aspect ratio bucketing.',
            recommendations: {
                prototyping: '256-512',
                production: '512-1024',
                maximum_quality: '1024+'
            }
        }
    ]
};
