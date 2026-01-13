# Neural Network Playground

<div align="center">

![Neural Network Playground](https://img.shields.io/badge/Deep%20Learning-Visualized-6366f1?style=for-the-badge&logo=tensorflow&logoColor=white)
![Vanilla JS](https://img.shields.io/badge/Vanilla-JavaScript-f7df1e?style=for-the-badge&logo=javascript&logoColor=black)
![No Dependencies](https://img.shields.io/badge/Zero-Dependencies-10b981?style=for-the-badge)
![Educational](https://img.shields.io/badge/Learn-By%20Doing-8b5cf6?style=for-the-badge)

**An interactive deep learning visualization platform built entirely from scratch.**

*Build, train, and visualize neural networks in real-time with stunning animations and intuitive controls.*

[**Live Demo**](https://dlmastery.github.io/neural-network-playground/) · [Features](#features) · [How It's Built](#how-its-built) · [Architecture](#architecture) · [Contributing](#contributing)

---

</div>

## Overview

Neural Network Playground is an educational tool that makes deep learning concepts accessible through beautiful, interactive visualizations. Watch neurons fire, see gradients flow, and understand how neural networks learn — all in your browser with zero dependencies.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│     ●───●───●                    ╭─────────────────╮                        │
│    /│\ /│\ /│\      TRAIN       │  Decision       │       Loss: 0.023      │
│   ● ● ● ● ● ● ●   ────────►    │  Boundary       │       Accuracy: 98%    │
│    \│/ \│/ \│/                  │  Visualization  │                        │
│     ●───●───●                    ╰─────────────────╯                        │
│                                                                             │
│   Neural Network                 Live Updates              Metrics          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

### 🧠 Four Interactive Tabs

| Tab | Description | Key Concepts |
|-----|-------------|--------------|
| **Neural Network** | Classic feedforward networks with real-time decision boundary visualization | Backpropagation, Activation Functions, Gradient Descent |
| **CNN** | Convolutional Neural Networks with layer-by-layer feature map visualization | Convolutions, Pooling, Feature Hierarchies |
| **Transformer** | Attention mechanism visualization with interactive token analysis | Self-Attention, Positional Encoding, Multi-Head Attention |
| **GNN** | Graph Neural Networks with message passing animation | Graph Convolution, Node Classification, Neighborhood Aggregation |

### ✨ Visual Highlights

- **Real-time Training** — Watch the network learn with live loss curves and accuracy metrics
- **Decision Boundaries** — See classification regions update as training progresses
- **Animated Data Flow** — Visualize forward and backward passes through the network
- **Interactive Graphs** — Click, drag, and explore network architectures
- **Glassmorphism UI** — Modern, beautiful interface with smooth animations
- **Dark Theme** — Easy on the eyes for extended learning sessions

## Getting Started

### Quick Start

```bash
# Clone the repository
git clone https://github.com/dlmastery/neural-network-playground.git

# Navigate to the project
cd neural-network-playground

# Start a local server (choose one)
python -m http.server 8080
# or
npx serve .
# or
php -S localhost:8080

# Open in browser
open http://localhost:8080
```

### No Build Required

This project uses vanilla JavaScript with ES6 modules. No npm install, no webpack, no transpilation — just open and run.

---

## How It's Built

This playground is built **entirely from scratch** — no TensorFlow, no PyTorch, no React. Every matrix multiplication, every gradient calculation, every pixel rendered is hand-crafted in vanilla JavaScript. Here's how:

### The Math Engine (`js/core/`)

At the heart of the playground is a custom tensor library that handles all numerical computations:

```javascript
// tensor.js - Matrix operations from scratch
class Tensor {
    constructor(data, shape) {
        this.data = new Float32Array(data);  // Typed arrays for performance
        this.shape = shape;
    }

    matmul(other) { /* O(n³) matrix multiplication */ }
    add(other)    { /* Element-wise addition */ }
    relu()        { /* max(0, x) activation */ }
    softmax()     { /* Normalized exponentials */ }
}
```

**Why Float32Array?** JavaScript's typed arrays provide predictable memory layout and faster numerical operations compared to regular arrays — crucial when computing thousands of matrix multiplications per second during training.

### Neural Network Layers (`js/core/layer.js`)

Each layer type implements forward and backward passes:

```javascript
class DenseLayer {
    forward(input) {
        // z = Wx + b
        this.input = input;  // Cache for backprop
        this.z = this.weights.matmul(input).add(this.bias);
        return this.activation.forward(this.z);
    }

    backward(gradOutput) {
        // Chain rule: ∂L/∂W = ∂L/∂a · ∂a/∂z · xᵀ
        const gradZ = this.activation.backward(gradOutput);
        this.gradWeights = gradZ.matmul(this.input.transpose());
        this.gradBias = gradZ.sum(axis=1);
        return this.weights.transpose().matmul(gradZ);
    }
}
```

### Canvas Rendering (`js/visualization/`)

All visualizations use the HTML5 Canvas API directly — no D3.js, no Chart.js:

```javascript
// Decision boundary: Sample a grid, color by prediction
for (let x = 0; x < width; x += resolution) {
    for (let y = 0; y < height; y += resolution) {
        const input = screenToData(x, y);
        const prediction = network.forward(input);
        const color = predictionToColor(prediction);
        ctx.fillStyle = color;
        ctx.fillRect(x, y, resolution, resolution);
    }
}
```

**60fps animations** are achieved using `requestAnimationFrame` with careful batching of DOM updates and canvas redraws.

### CNN Implementation (`js/cnn/`)

The CNN tab implements convolutions from first principles:

```javascript
// 2D Convolution - sliding window over input
function convolve2D(input, kernel, stride = 1) {
    const output = [];
    for (let y = 0; y <= inputH - kernelH; y += stride) {
        for (let x = 0; x <= inputW - kernelW; x += stride) {
            let sum = 0;
            // Element-wise multiply and sum
            for (let ky = 0; ky < kernelH; ky++) {
                for (let kx = 0; kx < kernelW; kx++) {
                    sum += input[y + ky][x + kx] * kernel[ky][kx];
                }
            }
            output.push(sum);
        }
    }
    return output;
}
```

### Transformer Attention (`js/transformer/`)

Self-attention computed step-by-step:

```javascript
function selfAttention(Q, K, V) {
    // 1. Compute attention scores: QKᵀ
    const scores = Q.matmul(K.transpose());

    // 2. Scale by √dₖ to prevent softmax saturation
    const scaled = scores.scale(1 / Math.sqrt(keyDim));

    // 3. Softmax for attention weights
    const weights = softmax(scaled);  // Each row sums to 1

    // 4. Weighted sum of values
    return weights.matmul(V);
}
```

### Graph Neural Networks (`js/gnn/`)

GNN message passing aggregates neighbor features:

```javascript
function gcnLayer(H, W, adjNormalized) {
    // 1. Transform node features: HW
    const transformed = H.matmul(W);

    // 2. Aggregate from neighbors: ÃHW
    // adjNormalized = D^(-1/2) * A * D^(-1/2)
    const aggregated = adjNormalized.matmul(transformed);

    // 3. Apply activation
    return relu(aggregated);
}
```

**Force-directed layout** uses spring physics simulation:
- Nodes repel each other (Coulomb's law)
- Edges act as springs (Hooke's law)
- Velocity damping for stability

### Training Loop

Training runs in the main thread with `requestAnimationFrame` for smooth UI updates:

```javascript
function trainStep() {
    // Forward pass
    const predictions = network.forward(batchInputs);

    // Compute loss
    const loss = crossEntropyLoss(predictions, batchLabels);

    // Backward pass
    network.backward(loss.gradient);

    // Update weights
    optimizer.step(network.parameters);

    // Update visualizations
    updateLossChart(loss.value);
    updateDecisionBoundary();

    // Schedule next step
    if (training) requestAnimationFrame(trainStep);
}
```

### Performance Optimizations

| Technique | Where Used | Impact |
|-----------|------------|--------|
| `Float32Array` | Tensor operations | 3-5x faster than `Array` |
| `requestAnimationFrame` | All animations | Smooth 60fps, battery-efficient |
| Canvas batching | Network visualization | Reduced draw calls |
| Lazy tab loading | Tab initialization | Faster initial load |
| Object pooling | Particle system | Reduced GC pauses |

### Why Vanilla JavaScript?

Building from scratch serves the educational mission:

1. **Transparency** — No black-box library calls; every operation is visible
2. **Understanding** — You can trace a gradient from loss to weight update
3. **Portability** — Works offline, no CDN dependencies
4. **Performance** — No framework overhead; direct DOM/Canvas access
5. **Learning** — The code itself teaches how neural networks work

## Architecture

```
neural-network-playground/
├── index.html                 # Single-page application entry
├── css/
│   ├── main.css              # Global styles & design system
│   ├── components.css        # UI component library
│   ├── animations.css        # Keyframe animations
│   ├── cnn.css              # CNN tab styles
│   ├── transformer.css      # Transformer tab styles
│   └── gnn.css              # GNN tab styles
├── js/
│   ├── main.js              # App orchestration & tab management
│   ├── core/
│   │   ├── tensor.js        # Matrix operations (from scratch!)
│   │   ├── layer.js         # Dense, Conv, Activation layers
│   │   ├── network.js       # Neural network class
│   │   ├── optimizer.js     # SGD, Adam, RMSprop
│   │   └── loss.js          # MSE, Cross-entropy
│   ├── visualization/
│   │   ├── canvas.js        # Canvas utilities
│   │   ├── network-viz.js   # Network graph rendering
│   │   └── data-viz.js      # Decision boundary heatmaps
│   ├── cnn/
│   │   ├── cnn-model.js     # Convolutional network
│   │   ├── cnn-viz.js       # Feature map visualization
│   │   └── cnn-app.js       # CNN tab controller
│   ├── transformer/
│   │   ├── pico-transformer.js  # Mini transformer implementation
│   │   ├── transformer-viz.js   # Attention visualization
│   │   └── transformer-app.js   # Transformer tab controller
│   └── gnn/
│       ├── gnn-dataset.js   # Karate Club & graph utilities
│       ├── gnn-model.js     # GCN & GAT implementations
│       ├── gnn-viz.js       # Force-directed graph layout
│       └── gnn-app.js       # GNN tab controller
└── README.md
```

## Tab Deep Dives

### Neural Network Tab

The classic playground experience. Build feedforward networks and watch them learn.

**Features:**
- Add/remove layers and neurons with clicks
- Choose activation functions (ReLU, Sigmoid, Tanh, Softmax)
- Select optimizers (SGD, Adam, RMSprop)
- Pick datasets (Circle, XOR, Spiral, Clusters)
- Watch decision boundaries evolve in real-time

**Educational Concepts:**
- Forward propagation: `z = Wx + b`, `a = σ(z)`
- Backpropagation and gradient computation
- Learning rate and its effects
- Overfitting and regularization

---

### CNN Tab

Explore how convolutional networks see the world.

**Features:**
- Draw digits on an interactive canvas
- Watch activations propagate through conv layers
- Visualize learned filters and feature maps
- See pooling reduce spatial dimensions
- Understand the feature hierarchy

**Educational Concepts:**
- Convolution operations and kernel sliding
- Feature detection (edges → shapes → objects)
- Spatial invariance through pooling
- Fully connected classification head

---

### Transformer Tab

Demystify the architecture powering modern AI.

**Features:**
- Enter custom text for analysis
- Visualize attention patterns between tokens
- See how self-attention weights form
- Explore positional encoding effects
- Watch multi-head attention in action

**Educational Concepts:**
- Query, Key, Value projections
- Scaled dot-product attention
- Why attention enables long-range dependencies
- Positional encoding necessity

---

### GNN Tab

Learn how neural networks process graph-structured data.

**Features:**
- Interactive Karate Club dataset (classic benchmark)
- Force-directed graph layout with physics simulation
- Animated message passing between nodes
- 2D embedding projection showing learned representations
- Switch between GCN and GAT architectures
- Click nodes to inspect features and neighbors

**Educational Concepts:**
- Message passing paradigm
- Neighborhood aggregation
- Graph convolution: `H' = σ(D⁻¹AHW)`
- Attention-weighted aggregation (GAT)
- Node classification task

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Vanilla JavaScript** | Core logic, no framework overhead |
| **ES6 Modules** | Clean code organization |
| **HTML5 Canvas** | All visualizations rendered natively |
| **CSS3** | Glassmorphism, animations, responsive design |
| **Web Workers** | Background training (neural network tab) |

### Why No Dependencies?

1. **Educational** — See exactly how everything works
2. **Lightweight** — Instant load, no bundle bloat
3. **Portable** — Works anywhere with a browser
4. **Transparent** — No magic, just math

## Design System

### Color Palette

```css
--color-primary:    #6366f1;  /* Indigo - Interactive elements */
--color-secondary:  #8b5cf6;  /* Purple - Accents */
--color-success:    #10b981;  /* Emerald - Positive metrics */
--color-warning:    #f59e0b;  /* Amber - Caution states */
--color-error:      #ef4444;  /* Red - Errors, negative class */
--color-bg:         #0f172a;  /* Dark slate - Background */
```

### Visual Effects

- **Glassmorphism** panels with backdrop blur
- **Gradient** borders and glowing accents
- **60fps** smooth animations
- **Particle** background system
- **Flowing** data animations along network edges

## Math Reference

### Feedforward Networks

```
Forward:  z = Wx + b,  a = activation(z)
Backward: ∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
Update:   W = W - η · ∂L/∂W
```

### Convolution

```
Output[i,j] = Σₘ Σₙ Input[i+m, j+n] · Kernel[m,n]
```

### Self-Attention

```
Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V
```

### Graph Convolution

```
H⁽ˡ⁺¹⁾ = σ(D̃⁻½ Ã D̃⁻½ H⁽ˡ⁾ W⁽ˡ⁾)
where Ã = A + I (adjacency + self-loops)
```

## Browser Support

| Browser | Version |
|---------|---------|
| Chrome/Edge | 90+ |
| Firefox | 90+ |
| Safari | 15+ |

Uses native ES modules — no transpilation needed.

## Contributing

Contributions are welcome! Here are some ideas:

### Feature Ideas

- [ ] RNN/LSTM tab with sequence visualization
- [ ] GAN tab with generator/discriminator dynamics
- [ ] Diffusion model visualization
- [ ] More datasets (MNIST subset, custom uploads)
- [ ] Export trained models
- [ ] Shareable playground configurations

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- ES6+ JavaScript with JSDoc comments
- BEM-like CSS naming (`.component__element--modifier`)
- Semantic HTML with ARIA labels
- No external dependencies (keep it vanilla!)

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- **Zachary's Karate Club** — Classic GNN benchmark dataset
- **TensorFlow Playground** — Inspiration for the original concept
- **3Blue1Brown** — For making neural network math beautiful
- **The Deep Learning Community** — For endless learning resources

---

<div align="center">

**Built with passion for education**

*Making deep learning intuitive, one visualization at a time.*

⭐ Star this repo if you find it helpful!

</div>
