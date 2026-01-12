# Neural Network Playground

A stunning, interactive deep learning visualization website built with vanilla JavaScript. Users can build, train, and visualize neural networks in real-time with beautiful animations and intuitive controls.

## Project Vision

Create an educational yet visually spectacular neural network playground that makes deep learning concepts accessible through interactive visualization. Think TensorFlow Playground meets modern creative coding aesthetics.

## Tech Stack

- **Vanilla JavaScript** (ES6+ modules) - No frameworks, pure JS for learning
- **HTML5 Canvas** - Network visualization and animations
- **CSS3** - Modern styling with animations, glassmorphism, gradients
- **Web Workers** - Background training without UI blocking
- **No external dependencies** - Everything built from scratch

## Architecture

```
deeplearning/
├── index.html              # Main entry point
├── css/
│   ├── main.css            # Global styles, variables, themes
│   ├── components.css      # UI component styles
│   └── animations.css      # Keyframes and transitions
├── js/
│   ├── main.js             # App initialization and orchestration
│   ├── core/
│   │   ├── tensor.js       # Matrix operations, tensor math
│   │   ├── layer.js        # Dense, activation layers
│   │   ├── network.js      # Neural network class
│   │   ├── optimizer.js    # SGD, Adam, RMSprop
│   │   └── loss.js         # MSE, cross-entropy
│   ├── visualization/
│   │   ├── canvas.js       # Canvas setup and utilities
│   │   ├── network-viz.js  # Neural network graph rendering
│   │   ├── data-viz.js     # Dataset and decision boundary viz
│   │   └── particles.js    # Background particle effects
│   ├── ui/
│   │   ├── controls.js     # Sliders, buttons, dropdowns
│   │   ├── sidebar.js      # Network architecture panel
│   │   └── metrics.js      # Loss/accuracy charts
│   ├── datasets/
│   │   ├── generators.js   # Circle, XOR, spiral, clusters
│   │   └── preprocessor.js # Normalization, train/test split
│   └── workers/
│       └── trainer.js      # Web Worker for training loop
├── assets/
│   └── fonts/              # Custom fonts if needed
└── CLAUDE.md               # This file
```

## Design System

### Color Palette
- **Primary**: `#6366f1` (Indigo) - Interactive elements
- **Secondary**: `#8b5cf6` (Purple) - Accents
- **Success**: `#10b981` (Emerald) - Positive metrics
- **Warning**: `#f59e0b` (Amber) - Caution states
- **Error**: `#ef4444` (Red) - Errors, negative class
- **Background**: `#0f172a` to `#1e293b` (Dark slate gradient)
- **Surface**: `rgba(255, 255, 255, 0.05)` - Glass panels
- **Text**: `#f8fafc` (primary), `#94a3b8` (secondary)

### Visual Effects
- Glassmorphism panels with `backdrop-filter: blur()`
- Gradient borders and glowing accents
- Smooth 60fps animations
- Particle background system
- Flowing data animations along network edges

### Typography
- System font stack for performance
- Monospace for metrics: `'JetBrains Mono', 'Fira Code', monospace`

## Core Features

### 1. Interactive Network Builder
- Drag-and-drop layer creation
- Click to add/remove neurons
- Visual connections with weight coloring (blue=positive, red=negative)
- Animated forward/backward pass visualization

### 2. Dataset Selection
- **Circle**: Points inside/outside circle
- **XOR**: Classic XOR problem
- **Spiral**: Two interleaved spirals
- **Clusters**: Gaussian blobs
- **Custom**: Click to add points

### 3. Training Controls
- Play/pause/step training
- Learning rate slider (0.001 - 1.0)
- Batch size selection
- Optimizer choice (SGD, Adam)
- Regularization (L1, L2, Dropout)

### 4. Real-time Visualization
- Decision boundary heatmap updates live
- Loss/accuracy curves
- Weight/gradient histograms
- Neuron activation patterns

### 5. Activation Functions
- ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, Swish

## Code Conventions

### JavaScript
- Use ES6 modules (`import`/`export`)
- Classes for core components (Network, Layer, Tensor)
- Pure functions for math operations
- Descriptive variable names (`learningRate` not `lr`)
- JSDoc comments for public APIs

### CSS
- CSS custom properties for theming
- BEM-like naming: `.network-viz__node--active`
- Mobile-first responsive design
- Use `clamp()` for fluid typography

### HTML
- Semantic elements (`<main>`, `<section>`, `<aside>`)
- Accessible: ARIA labels, keyboard navigation
- No inline styles or scripts

## Performance Guidelines

- Use `requestAnimationFrame` for all animations
- Batch DOM updates
- Use `will-change` sparingly for animated elements
- Offload training to Web Workers
- Use typed arrays (`Float32Array`) for tensor operations
- Debounce/throttle expensive UI updates

## Implementation Priorities

1. **Phase 1**: Core math (Tensor class with matrix ops)
2. **Phase 2**: Basic network (Layer, Network, forward pass)
3. **Phase 3**: Canvas visualization (static network display)
4. **Phase 4**: Training loop with backprop
5. **Phase 5**: UI controls and interactivity
6. **Phase 6**: Decision boundary visualization
7. **Phase 7**: Polish (animations, particles, effects)

## Neural Network Math Reference

### Forward Pass
```
z = W * x + b
a = activation(z)
```

### Backpropagation
```
dL/dW = dL/da * da/dz * dz/dW
dL/db = dL/da * da/dz
```

### Gradient Descent Update
```
W = W - learning_rate * dL/dW
b = b - learning_rate * dL/db
```

## Testing Approach

- Manual testing in browser console
- Visual regression testing (screenshot comparison)
- Test with known datasets (XOR should converge)

## Browser Support

Target modern browsers only:
- Chrome/Edge 90+
- Firefox 90+
- Safari 15+

Use native ES modules, no transpilation needed.
