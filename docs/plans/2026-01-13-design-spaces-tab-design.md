# Design Spaces Tab Design

## Overview

Add a 6th tab to the Neural Network Playground for **Design Spaces** - an interactive educational reference covering architecture design parameters for modern deep learning models. Serves as a practitioner's cheat-sheet with best practice guidance.

## Goals

- Provide quick reference for ~10 key design parameters per architecture
- Cover 10 modern architectures comprehensively
- Interactive UI with parameter cards, sliders, and tooltips
- Educational only - no simulations, pure reference material
- Match visual quality and polish of existing tabs

---

## Architectures Covered (10 Total)

1. **Transformers** - Attention mechanisms, positional encodings
2. **Large Language Models (LLMs)** - Scaling laws, context length, MoE
3. **Diffusion Models** - Noise schedules, U-Net design, sampling
4. **Graph Neural Networks (GNNs)** - Message passing, over-smoothing
5. **Convolutional Neural Networks (CNNs)** - Kernel sizes, pooling, depth
6. **RNNs & LSTMs** - Hidden state, gates, bidirectionality
7. **State Space Models (SSMs/Mamba)** - Selective mechanisms, state expansion
8. **Vision Transformers (ViTs)** - Patch size, conv stems, multi-stage
9. **Variational Autoencoders (VAEs)** - Latent dim, KL weight, architecture
10. **Generative Adversarial Networks (GANs)** - Discriminator design, training tricks

---

## Main Tab Interface

### Layout (Two Columns)

**Left Sidebar - Architecture Navigator**
- Vertical list of 10 architectures with icons
- Active architecture highlighted
- Expandable sections showing key stats:
  - Parameter count range
  - Typical use cases
  - Year introduced

**Right Main Area - Parameter Cards**
- Grid of parameter cards (2-3 columns)
- Each card shows:
  - Parameter name
  - Interactive control (slider/dropdown/toggle)
  - Current value display
  - Best practice tip (collapsible)
  - Trade-off indicator (compute vs quality)
- "Copy Config" button - exports parameter summary
- "Reset to Defaults" button

**Top Controls Bar**
- Search box - filter parameters across architectures
- View toggle: Cards | Table | Comparison
- "Explain" button - opens detailed explainer modal

---

## Parameter Definitions by Architecture

### 1. Transformers

| Parameter | Type | Options/Range | Default | Best Practice |
|-----------|------|---------------|---------|---------------|
| Layers (L) | Slider | 1-96 | 12 | 6-12 for most tasks; scale with data |
| Model Dim (d_model) | Slider | 64-4096 | 512 | Power of 2; 512-1024 common |
| Attention Heads | Slider | 1-64 | 8 | d_model/64 is typical |
| Head Dim (d_k) | Computed | d_model/heads | 64 | 64 is optimal for hardware |
| FFN Dim | Slider | 256-16384 | 2048 | 4× d_model standard |
| Attention Type | Dropdown | MHA, MQA, GQA | MHA | GQA balances efficiency/quality |
| Positional Encoding | Dropdown | Sinusoidal, Learned, RoPE, ALiBi | RoPE | RoPE for length generalization |
| Dropout | Slider | 0-0.5 | 0.1 | Lower for larger models |
| Layer Norm | Dropdown | Pre-LN, Post-LN | Pre-LN | Pre-LN more stable |
| Activation | Dropdown | ReLU, GELU, SwiGLU | SwiGLU | SwiGLU for modern LLMs |

### 2. Large Language Models (LLMs)

| Parameter | Type | Options/Range | Default | Best Practice |
|-----------|------|---------------|---------|---------------|
| Total Params | Display | 125M-405B | - | Follow Chinchilla scaling |
| Context Length | Slider | 512-128K | 4096 | Longer = more memory; use RoPE scaling |
| Vocab Size | Slider | 32K-256K | 50K | Larger for multilingual |
| Tokenizer | Dropdown | BPE, SentencePiece, Tiktoken | BPE | BPE with byte fallback |
| Architecture | Dropdown | Dense, MoE | Dense | MoE for >70B params |
| Expert Count (MoE) | Slider | 8-128 | 8 | 8 experts is common |
| Top-K Routing (MoE) | Slider | 1-4 | 2 | 2 experts per token |
| KV Cache | Dropdown | Full, GQA, MQA, Paged | GQA | GQA for efficient inference |
| Tie Embeddings | Toggle | Yes/No | Yes | Reduces params, helps small models |
| Compute Budget | Display | FLOPs | - | 20× params in tokens (Chinchilla) |

### 3. Diffusion Models

| Parameter | Type | Options/Range | Default | Best Practice |
|-----------|------|---------------|---------|---------------|
| Timesteps (T) | Slider | 100-1000 | 1000 | 1000 for training, fewer for inference |
| Noise Schedule | Dropdown | Linear, Cosine, Scaled Linear | Cosine | Cosine preserves more signal |
| Beta Start | Slider | 0.0001-0.001 | 0.0001 | Lower = less noise initially |
| Beta End | Slider | 0.01-0.1 | 0.02 | Higher = more noise finally |
| Sampler | Dropdown | DDPM, DDIM, Euler, DPM++ | DDIM | DDIM for fast inference |
| Inference Steps | Slider | 10-100 | 20 | 20-50 for quality/speed balance |
| Guidance Scale | Slider | 1-20 | 7.5 | 7-9 typical; higher = more faithful |
| U-Net Channels | Slider | 64-320 | 128 | Scale with image resolution |
| Attention Res | Multi-select | 32, 16, 8 | 16, 8 | Lower res = more global attention |
| Conditioning | Dropdown | Class, Text, Image | Text | Text via CLIP embeddings |

### 4. Graph Neural Networks (GNNs)

| Parameter | Type | Options/Range | Default | Best Practice |
|-----------|------|---------------|---------|---------------|
| Layers | Slider | 2-8 | 3 | 2-3 to avoid over-smoothing |
| Hidden Dim | Slider | 32-512 | 128 | Depends on graph complexity |
| Aggregation | Dropdown | Mean, Sum, Max, Attention | Mean | Sum for counting, Mean for averaging |
| Message Passing | Dropdown | GCN, GAT, GraphSAGE, GIN | GCN | GIN for expressivity |
| Skip Connections | Dropdown | None, Residual, Dense, JK | Residual | JK for variable receptive fields |
| Normalization | Dropdown | None, BatchNorm, LayerNorm, PairNorm | BatchNorm | PairNorm prevents over-smoothing |
| Dropout | Slider | 0-0.5 | 0.5 | GNNs need higher dropout |
| Edge Dropout | Slider | 0-0.5 | 0.2 | DropEdge helps generalization |
| Readout | Dropdown | Mean, Sum, Set2Set, Virtual Node | Mean | Virtual node for global context |
| Positional Encoding | Dropdown | None, Laplacian, Random Walk | Laplacian | For distinguishing symmetric structures |

### 5. Convolutional Neural Networks (CNNs)

| Parameter | Type | Options/Range | Default | Best Practice |
|-----------|------|---------------|---------|---------------|
| Kernel Size | Dropdown | 3×3, 5×5, 7×7 | 3×3 | 3×3 most efficient (VGG insight) |
| Depth | Slider | 5-152 | 50 | ResNet-50 is versatile baseline |
| Width Multiplier | Slider | 0.25-2.0 | 1.0 | Reduce for mobile (MobileNet) |
| Pooling | Dropdown | Max, Avg, Strided Conv | Max | Strided conv more flexible |
| Skip Connections | Dropdown | None, Residual, Dense | Residual | Essential for deep networks |
| Normalization | Dropdown | BatchNorm, LayerNorm, GroupNorm | BatchNorm | GroupNorm for small batches |
| Activation | Dropdown | ReLU, LeakyReLU, SiLU, Mish | ReLU | SiLU for EfficientNet-style |
| Stem | Dropdown | 7×7 stride 2, Patchify, Conv Stack | 7×7 | Patchify for ConvNeXt |
| SE Blocks | Toggle | Yes/No | Yes | Squeeze-and-excite helps |
| Depthwise Separable | Toggle | Yes/No | No | Yes for mobile efficiency |

### 6. RNNs & LSTMs

| Parameter | Type | Options/Range | Default | Best Practice |
|-----------|------|---------------|---------|---------------|
| Hidden Size | Slider | 64-2048 | 512 | Larger = more capacity |
| Layers | Slider | 1-8 | 2 | 2-3 typical; diminishing returns |
| Cell Type | Dropdown | Vanilla, LSTM, GRU | LSTM | GRU simpler, LSTM more expressive |
| Bidirectional | Toggle | Yes/No | Yes | Yes for non-causal tasks |
| Dropout | Slider | 0-0.5 | 0.2 | Between layers |
| Input Dropout | Slider | 0-0.5 | 0.1 | On input embeddings |
| Residual | Toggle | Yes/No | Yes | Helps deeper networks |
| Layer Norm | Toggle | Yes/No | Yes | Stabilizes training |
| Gradient Clipping | Slider | 0.1-10 | 1.0 | Essential for training |
| Truncated BPTT | Slider | 35-200 | 70 | Longer = more context, more memory |

### 7. State Space Models (SSMs/Mamba)

| Parameter | Type | Options/Range | Default | Best Practice |
|-----------|------|---------------|---------|---------------|
| Model Dim (D) | Slider | 256-4096 | 1024 | Scale with task complexity |
| State Expansion (N) | Slider | 16-64 | 16 | Larger = more expressivity |
| Conv Width (d_conv) | Slider | 2-8 | 4 | 4 is standard from Mamba paper |
| Expansion Factor | Slider | 1-4 | 2 | Inner dim = 2× model_dim |
| Layers | Slider | 12-64 | 24 | Similar scaling to Transformers |
| Discretization | Dropdown | ZOH, Bilinear | ZOH | Zero-order hold is standard |
| Initialization | Dropdown | S4D, Random | S4D | S4D helps stability |
| Activation | Dropdown | SiLU, GELU | SiLU | SiLU standard for Mamba |
| Normalization | Dropdown | RMSNorm, LayerNorm | RMSNorm | RMSNorm before each block |
| Bidirectional | Toggle | Yes/No | No | Yes for non-causal tasks |

### 8. Vision Transformers (ViTs)

| Parameter | Type | Options/Range | Default | Best Practice |
|-----------|------|---------------|---------|---------------|
| Patch Size | Dropdown | 4, 8, 14, 16, 32 | 16 | Smaller = more tokens, higher compute |
| Image Size | Slider | 224-512 | 224 | 224 standard, 384/512 for fine-tuning |
| Embed Dim | Slider | 192-1024 | 768 | ViT-B: 768, ViT-L: 1024 |
| Depth | Slider | 12-32 | 12 | ViT-B: 12, ViT-L: 24 |
| Heads | Slider | 3-16 | 12 | embed_dim / 64 |
| MLP Ratio | Slider | 2-6 | 4 | FFN hidden = 4× embed_dim |
| Conv Stem | Toggle | Yes/No | No | Improves training stability |
| Class Token | Dropdown | CLS Token, GAP | CLS | GAP often simpler |
| Pos Encoding | Dropdown | Learned, Sinusoidal, 2D | Learned | 2D for variable resolutions |
| Multi-stage | Toggle | Yes/No | No | Yes for dense prediction (Swin) |

### 9. Variational Autoencoders (VAEs)

| Parameter | Type | Options/Range | Default | Best Practice |
|-----------|------|---------------|---------|---------------|
| Latent Dim | Slider | 2-512 | 128 | Higher = more capacity |
| Encoder Layers | Slider | 2-6 | 4 | Symmetric with decoder |
| Decoder Layers | Slider | 2-6 | 4 | Mirror encoder architecture |
| Hidden Dim | Slider | 64-1024 | 256 | Gradually increase to bottleneck |
| KL Weight (β) | Slider | 0.001-10 | 1.0 | <1 for reconstruction, >1 for disentangling |
| Annealing | Toggle | Yes/No | Yes | Gradually increase β during training |
| Prior | Dropdown | Standard Normal, Learned, VampPrior | Standard | VampPrior for complex data |
| Decoder Output | Dropdown | Gaussian, Bernoulli | Gaussian | Bernoulli for binary data |
| Free Bits | Slider | 0-2 | 0 | Prevents posterior collapse |
| Architecture | Dropdown | MLP, CNN, ResNet | CNN | CNN for images |

### 10. Generative Adversarial Networks (GANs)

| Parameter | Type | Options/Range | Default | Best Practice |
|-----------|------|---------------|---------|---------------|
| Latent Dim (z) | Slider | 64-512 | 128 | 128-256 typical |
| Generator Layers | Slider | 4-8 | 5 | Progressive growing helps |
| Discriminator Layers | Slider | 4-8 | 5 | Match generator depth |
| Base Channels | Slider | 32-128 | 64 | Double at each resolution |
| Normalization (G) | Dropdown | BatchNorm, InstanceNorm, None | BatchNorm | InstanceNorm for StyleGAN |
| Normalization (D) | Dropdown | None, SpectralNorm, LayerNorm | SpectralNorm | SpectralNorm stabilizes D |
| Loss | Dropdown | Vanilla, WGAN, WGAN-GP, Hinge | Hinge | Hinge loss stable and simple |
| Learning Rate (G) | Slider | 0.0001-0.001 | 0.0002 | Often lower than D |
| Learning Rate (D) | Slider | 0.0001-0.001 | 0.0002 | Match or slightly higher than G |
| D Updates per G | Slider | 1-5 | 1 | WGAN: 5, others: 1 |

---

## Explainer Modal (Per Architecture)

When user clicks "Explain" for any architecture, show modal with:

### Structure
1. **Overview** - What is this architecture? Key innovation
2. **Core Mechanism** - Animated diagram of key concept
3. **Design Trade-offs** - Interactive exploration
4. **Parameter Interactions** - How parameters affect each other
5. **When to Use** - Use case guidance
6. **Common Mistakes** - Pitfalls to avoid

### Interactive Elements
- Animated diagrams showing architecture concepts
- Sliders showing parameter trade-offs (compute vs quality)
- Comparison charts (e.g., attention types, normalizations)
- "Why this default?" expandable explanations

---

## Visual Design

### Colors (Design Spaces Accent)
- **Primary Accent:** `#14b8a6` (Teal) - Distinguishes from other tabs
- **Architecture Icons:** Unique color per architecture
- **Parameter Cards:** Glass panels with teal accents
- **Trade-off Indicators:** Red-Yellow-Green spectrum

### Architecture Color Scheme
| Architecture | Color | Icon |
|--------------|-------|------|
| Transformers | `#6366f1` (Indigo) | ⚡ |
| LLMs | `#8b5cf6` (Purple) | 🗣️ |
| Diffusion | `#ec4899` (Pink) | 🌊 |
| GNNs | `#10b981` (Emerald) | 🕸️ |
| CNNs | `#f59e0b` (Amber) | 🔲 |
| RNNs/LSTMs | `#ef4444` (Red) | ↩️ |
| SSMs/Mamba | `#06b6d4` (Cyan) | 〰️ |
| ViTs | `#a855f7` (Violet) | 🖼️ |
| VAEs | `#22c55e` (Green) | 🔄 |
| GANs | `#f97316` (Orange) | ⚔️ |

### Parameter Card Design
```
┌─────────────────────────────────────┐
│ ⚙️ Attention Heads          [?]    │
│ ─────────────────────────────────── │
│ [====|================] 8           │
│ ─────────────────────────────────── │
│ 💡 Best: d_model/64 (usually 64)   │
│ ⚡ Compute: Low  📊 Impact: Medium  │
└─────────────────────────────────────┘
```

### Animations
- Card hover: Subtle lift and glow
- Architecture switch: Smooth cross-fade
- Parameter change: Value pulse
- Tooltips: Fade in from pointer

### Responsive Design
- **≥1200px:** 3-column parameter grid
- **900-1199px:** 2-column grid, sidebar collapses
- **600-899px:** Single column, tabs for architectures
- **<600px:** Full-width cards, bottom nav for architectures

---

## File Structure

```
js/design-spaces/
├── design-spaces-app.js      # Main tab orchestration
├── design-spaces-data.js     # All parameter definitions
├── design-spaces-viz.js      # Parameter cards, architecture diagrams
├── design-spaces-explainer.js # Per-architecture explainer modals

css/
├── design-spaces.css         # All design-spaces-specific styles
```

---

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Create file structure
- [ ] Add design-spaces tab to index.html
- [ ] Create design-spaces.css with layout
- [ ] Implement design-spaces-app.js skeleton
- [ ] Create design-spaces-data.js with all parameters

### Phase 2: Architecture Navigator
- [ ] Implement sidebar with 10 architectures
- [ ] Add architecture icons and colors
- [ ] Implement architecture switching
- [ ] Add key stats display

### Phase 3: Parameter Cards
- [ ] Create reusable parameter card component
- [ ] Implement slider controls
- [ ] Implement dropdown controls
- [ ] Implement toggle controls
- [ ] Add best practice tooltips
- [ ] Add trade-off indicators

### Phase 4: Interactions
- [ ] Implement search/filter
- [ ] Add comparison view
- [ ] Implement "Copy Config" export
- [ ] Add "Reset to Defaults"

### Phase 5: Explainer Modals
- [ ] Create modal structure
- [ ] Implement overview animations
- [ ] Add interactive trade-off explorations
- [ ] Add "When to Use" guidance

### Phase 6: Polish
- [ ] Animations and transitions
- [ ] Responsive design
- [ ] Keyboard navigation
- [ ] Performance optimization

---

## Success Criteria

1. All 10 architectures covered with ~10 parameters each
2. Parameter cards are interactive and informative
3. Best practice guidance is clear and actionable
4. Works as a quick reference (find info in <10 seconds)
5. Matches visual quality of existing tabs
6. Responsive across all breakpoints
