# Diffusion Model Tab Design

## Overview

Add a 5th tab to the Neural Network Playground for **Diffusion Models** - the generative AI technique behind Stable Diffusion, DALL-E, etc. The tab will generate MNIST digits (28x28) using a tiny U-Net architecture with WebGPU as the default backend.

## Goals

- Teach diffusion concepts progressively (7 stages from intuition to advanced)
- Generate MNIST digits to connect with existing CNN tab
- Provide both pre-trained inference and optional training mode
- Use WebGPU for performance, with fallbacks

---

## Main Tab Interface

### Layout (Three Columns)

**Left Sidebar - Model & Controls**
- Model status indicator (loaded/training/generating)
- Mini U-Net architecture diagram
- Class selector (digits 0-9 for conditional generation)
- Generation controls:
  - Grid size: 1×1, 2×2, 3×3, 4×4
  - Inference steps: 10-50 slider
  - Guidance scale: 1.0-10.0
- "Train Your Own" toggle → reveals:
  - Learning rate input
  - Epochs slider
  - Train/Stop buttons
  - Loss display

**Center - Generation Grid**
- Primary 4×4 canvas grid (default)
- Each cell: 28×28 generated digit
- Click cell → select for detail view
- "Generate" button with loading state
- Progress bar during generation

**Right Sidebar - Denoising Preview**
- Selected digit's full denoising timeline
- Vertical filmstrip: t=T (noise) → t=0 (clean)
- Interactive scrubber to pause at any step
- Noise level indicator
- Seed display

**Top Controls Bar**
- Backend selector: WebGPU (default) | WebGL | CPU
- Status metrics: tokens/s, current step
- "Explain" button → opens explainer modal

---

## Explainer Modal - 7 Progressive Tabs

### Tab 1: What is Diffusion?
**Content:**
- Animated "ink drop in water" analogy
- Forward: digit → noise (spreading)
- Reverse: noise → digit (gathering)
- Side-by-side visualization

**Key Insight:** "Diffusion models learn to reverse the natural process of adding noise"

### Tab 2: Forward Process (Noising)
**Content:**
- Interactive slider: user adds noise to a digit
- Mathematical notation: q(xₜ|xₜ₋₁) = N(√(1-βₜ)xₜ₋₁, βₜI)
- Variance schedule visualization (linear ramp of βₜ)
- Show 10 timesteps from clean to pure noise

**Interactive:** Drag slider to control noise level on any digit

### Tab 3: Reverse Process (Denoising)
**Content:**
- Step-by-step denoising animation
- Overlay: predicted noise (pink) vs actual noise
- The reverse equation: pθ(xₜ₋₁|xₜ)
- Explanation of what the neural network predicts

**Key Insight:** "The network predicts the noise to subtract at each step"

### Tab 4: The U-Net Architecture
**Content:**
- Interactive U-Net diagram with:
  - Encoder path (downsampling)
  - Bottleneck
  - Decoder path (upsampling)
  - Skip connections highlighted
- Time embedding injection visualization
- Class embedding for conditional generation
- Click any block → see feature map shapes

**Interactive:** Hover layers to highlight data flow, click for details

### Tab 5: Noise Schedules
**Content:**
- Visual comparison: Linear vs Cosine schedules
- Graph showing βₜ values over timesteps
- Side-by-side generation with different schedules
- Explanation of why schedule matters

**Interactive:** Toggle between schedules, regenerate with same seed

### Tab 6: Sampling Methods
**Content:**
- DDPM (stochastic, 50 steps) vs DDIM (deterministic, 10 steps)
- Race visualization: both methods generating same digit
- Speed vs quality tradeoff explanation
- Mathematical difference (brief)

**Interactive:** Run both samplers side-by-side on same noise

### Tab 7: Guidance (Class-Conditional)
**Content:**
- Classifier-free guidance explanation
- Guidance scale slider: 1.0 → 10.0
- Visual: low guidance (diverse, fuzzy) → high guidance (sharp, less diverse)
- How unconditional and conditional predictions combine

**Interactive:** Adjust guidance scale, see results in real-time

---

## Technical Architecture

### File Structure
```
js/diffusion/
├── diffusion-model.js      # U-Net, scheduler, sampling algorithms
├── diffusion-dataset.js    # MNIST loading (reuse CNN loader)
├── diffusion-viz.js        # Canvas rendering (grid, timeline, U-Net)
├── diffusion-explainer.js  # 7-tab explainer interactions
├── diffusion-app.js        # Main tab orchestration, state management

css/
├── diffusion.css           # All diffusion-specific styles
```

### Tiny U-Net Architecture (~100K parameters)

```
Input: 28×28×1 + time_embedding + class_embedding

Encoder:
  - conv_in: 1 → 32 channels (28×28)
  - down_block_1: 32 → 64 channels + downsample (14×14)
  - down_block_2: 64 → 128 channels + downsample (7×7)

Bottleneck:
  - 128 channels at 7×7
  - Self-attention (optional, for quality)

Decoder:
  - up_block_1: 128+128 → 64 channels + upsample (14×14)  [skip from down_2]
  - up_block_2: 64+64 → 32 channels + upsample (28×28)    [skip from down_1]
  - conv_out: 32 → 1 channel (predicted noise)

Time Embedding:
  - Sinusoidal positional encoding (like transformers)
  - MLP: embed_dim → hidden → embed_dim
  - Added to each block's features

Class Embedding:
  - Learned embedding table: 10 classes × embed_dim
  - Added to time embedding
```

### WebGPU Implementation

**Inference (Primary Path):**
- Custom WGSL compute shaders for:
  - Conv2D operations
  - GroupNorm
  - SiLU activation
  - Upsample/Downsample
- No TensorFlow.js dependency for inference
- Pre-trained weights: ~400KB (quantized Float16)

**Training (Optional Mode):**
- Uses TensorFlow.js with WebGPU backend
- MSE loss on predicted vs actual noise
- Adam optimizer, configurable learning rate
- Batch size: 32
- Training data: 1000 MNIST samples (subset for browser)

**Fallback Chain:**
1. WebGPU (default, fastest)
2. WebGL via tfjs (broad support)
3. CPU/WASM (slowest, always works)

### Noise Schedule

```javascript
// Linear schedule (default)
const betaStart = 0.0001;
const betaEnd = 0.02;
const betas = linspace(betaStart, betaEnd, numTimesteps);

// Cosine schedule (optional)
const alphasCumprod = cosineSchedule(numTimesteps);
```

### Sampling Algorithms

**DDPM (Default):**
- 50 steps
- Stochastic (adds noise at each step)
- Higher quality, slower

**DDIM (Fast):**
- 10-20 steps
- Deterministic
- Same seed → same output
- Slightly lower quality

---

## Visual Design

### Colors
- **Diffusion Accent:** `#ec4899` (Pink) - distinguishes from other tabs
- **Noise Overlay:** `rgba(236, 72, 153, 0.5)` - predicted noise visualization
- **Timeline Gradient:** `#1e293b` (noisy) → `#f8fafc` (clean)
- **U-Net Blocks:**
  - Encoder: `#6366f1` (Indigo)
  - Decoder: `#8b5cf6` (Purple)
  - Skip connections: `#10b981` (Emerald)

### Animations
- **Generation Grid:** Cells fade in as each digit completes
- **Denoising Filmstrip:** Smooth vertical scroll during generation
- **U-Net Diagram:** Pulse animation showing data flow
- **Noise Particles:** Subtle floating particles (extends particles.js)

### Interactions
- **Hover digit:** Shows seed, class, generation time
- **Click digit:** Expands to full denoising timeline view
- **Drag timeline:** Scrub through denoising steps
- **Right-click digit:** Context menu (regenerate, copy seed, save)

### Responsive
- **< 1200px:** Grid reduces to 3×3
- **< 900px:** Grid reduces to 2×2, sidebars stack below
- **< 600px:** Single column, explainer tabs become horizontal scroll

---

## Educational Tooltips

Following CNN/Transformer pattern with `?` icons:

- **Timestep:** "The current position in the denoising process. t=T is pure noise, t=0 is the final image."
- **Guidance Scale:** "Controls how strongly the model follows the class label. Higher = more recognizable but less diverse."
- **Skip Connection:** "Passes earlier layer features directly to later layers, preserving fine details."
- **DDIM:** "Denoising Diffusion Implicit Models - a faster sampling method that's deterministic."

---

## Pre-trained Weights

**Training Setup (offline):**
- Full MNIST dataset (60K images)
- 100 epochs
- Cosine learning rate schedule
- Trained on GPU, exported to browser format

**Weight Format:**
- Float16 quantized for size
- ~400KB compressed
- Loaded via fetch on tab activation
- Cached in IndexedDB for subsequent visits

---

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Create file structure
- [ ] Add diffusion tab to index.html
- [ ] Create diffusion.css with basic layout
- [ ] Implement diffusion-app.js skeleton

### Phase 2: U-Net Model
- [ ] Implement Tiny U-Net in diffusion-model.js
- [ ] Add noise scheduler (linear + cosine)
- [ ] Implement DDPM sampling
- [ ] Implement DDIM sampling
- [ ] Add time/class embeddings

### Phase 3: Visualization
- [ ] Generation grid canvas
- [ ] Denoising timeline filmstrip
- [ ] U-Net architecture diagram
- [ ] Progress indicators

### Phase 4: Explainer Modal
- [ ] Tab 1: What is Diffusion
- [ ] Tab 2: Forward Process
- [ ] Tab 3: Reverse Process
- [ ] Tab 4: U-Net Architecture
- [ ] Tab 5: Noise Schedules
- [ ] Tab 6: Sampling Methods
- [ ] Tab 7: Guidance

### Phase 5: Training Mode
- [ ] Integrate tfjs for training
- [ ] Loss visualization
- [ ] Sample generation during training
- [ ] Save/load trained weights

### Phase 6: Polish
- [ ] Animations and transitions
- [ ] Educational tooltips
- [ ] Responsive design
- [ ] Performance optimization
- [ ] Pre-trained weight loading

---

## Success Criteria

1. Users can generate recognizable MNIST digits in < 3 seconds (WebGPU)
2. Explainer clearly teaches the 7 progressive concepts
3. Training mode shows visible improvement over 50+ epochs
4. Works across Chrome, Edge, Firefox (with appropriate fallbacks)
5. Matches visual quality and polish of existing tabs
