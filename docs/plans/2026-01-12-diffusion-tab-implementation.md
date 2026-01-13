# Diffusion Tab Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a 5th tab to Neural Network Playground that teaches diffusion models through interactive MNIST digit generation.

**Architecture:** Tiny U-Net (~100K params) generates 28x28 digits via iterative denoising. WebGPU compute shaders for inference, TensorFlow.js fallback for training mode. 7-tab progressive explainer teaches concepts from intuition to guidance.

**Tech Stack:** Vanilla JS (ES6 modules), WebGPU/WGSL shaders, HTML5 Canvas, CSS3 glassmorphism, TensorFlow.js (training only)

---

## Phase 1: Core Infrastructure

### Task 1.1: Add Tab Button to Header

**Files:**
- Modify: `index.html:57-62` (tab navigation)

**Step 1: Add diffusion tab button**

In `index.html`, locate the tab navigation section and add the Diffusion button after GNN:

```html
<!-- Tab Navigation -->
<div class="tab-nav">
    <button class="tab-btn tab-btn--active" data-tab="mlp">Neural Net</button>
    <button class="tab-btn" data-tab="cnn">CNN</button>
    <button class="tab-btn" data-tab="transformer">Transformer</button>
    <button class="tab-btn" data-tab="gnn">GNN</button>
    <button class="tab-btn" data-tab="diffusion">Diffusion</button>
</div>
```

**Step 2: Verify in browser**

Open `index.html` in browser. Confirm 5 tabs visible in header. Diffusion tab should appear but do nothing when clicked (panel doesn't exist yet).

**Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add diffusion tab button to navigation"
```

---

### Task 1.2: Create Empty Tab Panel Structure

**Files:**
- Modify: `index.html` (add after GNN panel, before transformer explainer modal ~line 1048)

**Step 1: Add diffusion panel skeleton**

Add this after the closing `</div><!-- End GNN Tab Panel -->`:

```html
<!-- Diffusion Tab Panel -->
<div class="tab-panel" id="diffusion-panel">
    <!-- Controls Bar -->
    <div class="controls-bar glass-panel diffusion-controls">
        <div class="control-section">
            <span class="control-section__label">Status</span>
            <span class="inline-metric__value" id="diffusion-status">Ready</span>
        </div>

        <div class="control-divider"></div>

        <div class="control-section">
            <div class="training-controls">
                <button class="btn btn--primary btn--large" id="diffusion-generate-btn">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 5v14M5 12h14"/>
                    </svg>
                    <span class="btn-text">Generate</span>
                </button>
                <button class="btn btn--accent" id="diffusion-explain-btn">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                        <line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                    <span class="btn-text">Explain</span>
                </button>
            </div>
        </div>

        <div class="control-divider"></div>

        <div class="control-section">
            <span class="control-section__label">Steps</span>
            <input type="range" class="slider slider--compact" id="diffusion-steps" min="10" max="50" step="5" value="20">
            <span class="control-section__value" id="diffusion-steps-value">20</span>
        </div>

        <div class="control-section">
            <span class="control-section__label">Guidance</span>
            <input type="range" class="slider slider--compact" id="diffusion-guidance" min="1" max="10" step="0.5" value="3">
            <span class="control-section__value" id="diffusion-guidance-value">3.0</span>
        </div>

        <div class="control-section">
            <span class="control-section__label">Class</span>
            <select class="select select--compact" id="diffusion-class">
                <option value="-1">Random</option>
                <option value="0">0</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
                <option value="6">6</option>
                <option value="7">7</option>
                <option value="8">8</option>
                <option value="9">9</option>
            </select>
        </div>

        <div class="control-divider"></div>

        <div class="control-section control-section--backend">
            <span class="control-section__label">Backend</span>
            <select class="select select--compact" id="diffusion-backend">
                <option value="webgpu" selected>WebGPU</option>
                <option value="webgl">WebGL</option>
                <option value="cpu">CPU</option>
            </select>
            <span class="backend-status" id="diffusion-backend-status"></span>
        </div>
    </div>

    <!-- Main Content -->
    <div class="main-content diffusion-main">
        <!-- Left Sidebar - Model Info -->
        <aside class="sidebar sidebar--left glass-panel diffusion-sidebar">
            <div class="sidebar__header">
                <h3>U-Net Model</h3>
            </div>
            <div class="sidebar__content">
                <div class="diffusion-model-info">
                    <div class="model-info-item">
                        <span class="model-info-label">Parameters</span>
                        <span class="model-info-value" id="diffusion-params">~100K</span>
                    </div>
                    <div class="model-info-item">
                        <span class="model-info-label">Input</span>
                        <span class="model-info-value">28×28×1</span>
                    </div>
                    <div class="model-info-item">
                        <span class="model-info-label">Timesteps</span>
                        <span class="model-info-value" id="diffusion-timesteps">1000</span>
                    </div>
                </div>

                <!-- Mini U-Net Diagram -->
                <div class="diffusion-arch-mini">
                    <canvas id="diffusion-arch-mini-canvas" width="200" height="300"></canvas>
                </div>

                <!-- Training Toggle -->
                <div class="diffusion-train-toggle">
                    <label class="toggle-label">
                        <input type="checkbox" id="diffusion-train-mode">
                        <span>Training Mode</span>
                    </label>
                </div>

                <!-- Training Controls (hidden by default) -->
                <div class="diffusion-train-controls" id="diffusion-train-controls" style="display: none;">
                    <div class="train-control">
                        <label>Learning Rate</label>
                        <input type="number" id="diffusion-lr" value="0.0001" step="0.00001" min="0.00001" max="0.01">
                    </div>
                    <div class="train-control">
                        <label>Epochs</label>
                        <input type="number" id="diffusion-epochs" value="50" min="1" max="200">
                    </div>
                    <div class="train-buttons">
                        <button class="btn btn--primary btn--sm" id="diffusion-train-btn">Train</button>
                        <button class="btn btn--secondary btn--sm" id="diffusion-stop-train-btn" disabled>Stop</button>
                    </div>
                    <div class="train-metrics">
                        <div class="metric-row">
                            <span>Epoch</span>
                            <span id="diffusion-train-epoch">0</span>
                        </div>
                        <div class="metric-row">
                            <span>Loss</span>
                            <span id="diffusion-train-loss">-</span>
                        </div>
                    </div>
                </div>
            </div>
        </aside>

        <!-- Center - Generation Grid -->
        <section class="visualization diffusion-visualization">
            <div class="viz-panel glass-panel diffusion-grid-panel">
                <div class="viz-panel__header">
                    <h3>Generated Digits</h3>
                    <span class="viz-panel__badge" id="diffusion-grid-info">4×4 Grid</span>
                </div>
                <div class="viz-panel__content">
                    <canvas id="diffusion-grid-canvas" width="448" height="448"></canvas>
                </div>
                <div class="viz-panel__footer">
                    <div class="diffusion-progress">
                        <div class="progress-track">
                            <div id="diffusion-progress-bar" class="progress-fill" style="width: 0%"></div>
                        </div>
                        <span id="diffusion-progress-text">Ready</span>
                    </div>
                </div>
            </div>
        </section>

        <!-- Right Sidebar - Denoising Preview -->
        <aside class="sidebar sidebar--right glass-panel diffusion-preview">
            <div class="sidebar__header">
                <h3>Denoising Timeline</h3>
            </div>
            <div class="sidebar__content">
                <p class="diffusion-hint" id="diffusion-preview-hint">Click a generated digit to see its denoising process</p>
                <div class="diffusion-timeline" id="diffusion-timeline">
                    <canvas id="diffusion-timeline-canvas" width="200" height="400"></canvas>
                </div>
                <div class="diffusion-scrubber" id="diffusion-scrubber" style="display: none;">
                    <input type="range" id="diffusion-time-scrubber" min="0" max="100" value="100">
                    <span id="diffusion-time-label">t=0</span>
                </div>
            </div>
        </aside>
    </div>
</div><!-- End Diffusion Tab Panel -->
```

**Step 2: Verify in browser**

Open browser, click Diffusion tab. Should see layout structure (unstyled). Console should have no errors.

**Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add diffusion tab panel HTML structure"
```

---

### Task 1.3: Create CSS File and Link It

**Files:**
- Create: `css/diffusion.css`
- Modify: `index.html:19` (add stylesheet link)

**Step 1: Create diffusion.css with basic styles**

Create `css/diffusion.css`:

```css
/* ==========================================
   DIFFUSION TAB STYLES
   ========================================== */

/* Accent color for diffusion tab */
:root {
    --diffusion-accent: #ec4899;
    --diffusion-accent-light: #f472b6;
    --diffusion-accent-dark: #db2777;
}

/* Controls Bar */
.diffusion-controls {
    display: flex;
    align-items: center;
    gap: 1rem;
}

/* Main Layout */
.diffusion-main {
    display: grid;
    grid-template-columns: 260px 1fr 240px;
    gap: 1rem;
    padding: 1rem;
    height: calc(100vh - 180px);
    overflow: hidden;
}

/* Left Sidebar */
.diffusion-sidebar {
    display: flex;
    flex-direction: column;
    overflow-y: auto;
}

.diffusion-model-info {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.diffusion-model-info .model-info-item {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 6px;
}

.diffusion-model-info .model-info-label {
    color: var(--text-secondary);
    font-size: 0.8rem;
}

.diffusion-model-info .model-info-value {
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 0.85rem;
}

/* Mini Architecture Diagram */
.diffusion-arch-mini {
    margin: 1rem 0;
    padding: 0.5rem;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 8px;
}

.diffusion-arch-mini canvas {
    width: 100%;
    height: auto;
}

/* Training Toggle */
.diffusion-train-toggle {
    margin: 1rem 0;
    padding: 0.75rem;
    background: rgba(236, 72, 153, 0.1);
    border: 1px solid rgba(236, 72, 153, 0.3);
    border-radius: 8px;
}

.diffusion-train-toggle .toggle-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    font-size: 0.9rem;
}

.diffusion-train-toggle input[type="checkbox"] {
    width: 18px;
    height: 18px;
    accent-color: var(--diffusion-accent);
}

/* Training Controls */
.diffusion-train-controls {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 1rem;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 8px;
    margin-top: 0.5rem;
}

.diffusion-train-controls .train-control {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.diffusion-train-controls .train-control label {
    font-size: 0.75rem;
    color: var(--text-secondary);
}

.diffusion-train-controls .train-control input {
    padding: 0.4rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    color: var(--text-primary);
    font-family: var(--font-mono);
}

.diffusion-train-controls .train-buttons {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
}

.diffusion-train-controls .train-metrics {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.diffusion-train-controls .metric-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
}

/* Center - Grid Panel */
.diffusion-visualization {
    display: flex;
    flex-direction: column;
    min-height: 0;
}

.diffusion-grid-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
}

.diffusion-grid-panel .viz-panel__content {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
    min-height: 0;
}

.diffusion-grid-panel canvas {
    max-width: 100%;
    max-height: 100%;
    image-rendering: pixelated;
    border-radius: 8px;
    background: rgba(0, 0, 0, 0.3);
}

/* Progress Bar */
.diffusion-progress {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.5rem 1rem;
}

.diffusion-progress .progress-track {
    flex: 1;
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    overflow: hidden;
}

.diffusion-progress .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--diffusion-accent), var(--diffusion-accent-light));
    border-radius: 3px;
    transition: width 0.1s ease;
}

#diffusion-progress-text {
    font-size: 0.8rem;
    color: var(--text-secondary);
    min-width: 80px;
    text-align: right;
}

/* Right Sidebar - Timeline */
.diffusion-preview {
    display: flex;
    flex-direction: column;
}

.diffusion-preview .sidebar__content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.diffusion-hint {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-align: center;
    padding: 1rem;
}

.diffusion-timeline {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    overflow-y: auto;
    padding: 0.5rem;
}

.diffusion-timeline canvas {
    border-radius: 8px;
    background: rgba(0, 0, 0, 0.2);
}

.diffusion-scrubber {
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
}

.diffusion-scrubber input[type="range"] {
    width: 100%;
    accent-color: var(--diffusion-accent);
}

#diffusion-time-label {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--diffusion-accent);
}

/* Backend Status */
.diffusion-controls .backend-status {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--color-warning);
}

.diffusion-controls .backend-status.active {
    background: var(--color-success);
    box-shadow: 0 0 8px var(--color-success);
}

/* Responsive */
@media (max-width: 1200px) {
    .diffusion-main {
        grid-template-columns: 220px 1fr 200px;
    }
}

@media (max-width: 900px) {
    .diffusion-main {
        grid-template-columns: 1fr;
        grid-template-rows: auto 1fr auto;
    }

    .diffusion-sidebar,
    .diffusion-preview {
        max-height: 200px;
    }
}
```

**Step 2: Link stylesheet in index.html**

Add after gnn.css link (~line 19):

```html
<link rel="stylesheet" href="css/diffusion.css">
```

**Step 3: Verify in browser**

Refresh page, click Diffusion tab. Layout should be properly styled with three columns, pink accent color visible on training toggle.

**Step 4: Commit**

```bash
git add css/diffusion.css index.html
git commit -m "feat: add diffusion tab CSS styles"
```

---

### Task 1.4: Create JavaScript Module Files

**Files:**
- Create: `js/diffusion/diffusion-app.js`
- Create: `js/diffusion/diffusion-model.js`
- Create: `js/diffusion/diffusion-viz.js`
- Modify: `js/main.js` (import diffusion module)

**Step 1: Create diffusion-app.js skeleton**

Create `js/diffusion/diffusion-app.js`:

```javascript
/**
 * Diffusion Tab Application
 * Orchestrates the diffusion model demo and explainer
 */

export class DiffusionApp {
    constructor() {
        this.isInitialized = false;
        this.isGenerating = false;
        this.selectedCell = null;

        // DOM Elements
        this.elements = {};

        // State
        this.config = {
            steps: 20,
            guidanceScale: 3.0,
            targetClass: -1, // -1 = random
            gridSize: 4,
            backend: 'webgpu'
        };

        // Generation history for timeline
        this.generationHistory = [];
    }

    /**
     * Initialize the diffusion tab
     */
    async init() {
        if (this.isInitialized) return;

        console.log('[Diffusion] Initializing...');

        // Cache DOM elements
        this.cacheElements();

        // Setup event listeners
        this.setupEventListeners();

        // Check WebGPU support
        await this.checkBackendSupport();

        this.isInitialized = true;
        console.log('[Diffusion] Initialized');
    }

    /**
     * Cache DOM element references
     */
    cacheElements() {
        this.elements = {
            // Controls
            generateBtn: document.getElementById('diffusion-generate-btn'),
            explainBtn: document.getElementById('diffusion-explain-btn'),
            stepsSlider: document.getElementById('diffusion-steps'),
            stepsValue: document.getElementById('diffusion-steps-value'),
            guidanceSlider: document.getElementById('diffusion-guidance'),
            guidanceValue: document.getElementById('diffusion-guidance-value'),
            classSelect: document.getElementById('diffusion-class'),
            backendSelect: document.getElementById('diffusion-backend'),
            backendStatus: document.getElementById('diffusion-backend-status'),
            statusText: document.getElementById('diffusion-status'),

            // Training
            trainModeToggle: document.getElementById('diffusion-train-mode'),
            trainControls: document.getElementById('diffusion-train-controls'),
            trainBtn: document.getElementById('diffusion-train-btn'),
            stopTrainBtn: document.getElementById('diffusion-stop-train-btn'),
            trainEpoch: document.getElementById('diffusion-train-epoch'),
            trainLoss: document.getElementById('diffusion-train-loss'),

            // Canvases
            gridCanvas: document.getElementById('diffusion-grid-canvas'),
            archCanvas: document.getElementById('diffusion-arch-mini-canvas'),
            timelineCanvas: document.getElementById('diffusion-timeline-canvas'),

            // Progress
            progressBar: document.getElementById('diffusion-progress-bar'),
            progressText: document.getElementById('diffusion-progress-text'),

            // Timeline
            previewHint: document.getElementById('diffusion-preview-hint'),
            timeline: document.getElementById('diffusion-timeline'),
            scrubber: document.getElementById('diffusion-scrubber'),
            timeScrubber: document.getElementById('diffusion-time-scrubber'),
            timeLabel: document.getElementById('diffusion-time-label')
        };
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Generate button
        this.elements.generateBtn?.addEventListener('click', () => this.generate());

        // Explain button
        this.elements.explainBtn?.addEventListener('click', () => this.openExplainer());

        // Steps slider
        this.elements.stepsSlider?.addEventListener('input', (e) => {
            this.config.steps = parseInt(e.target.value);
            this.elements.stepsValue.textContent = this.config.steps;
        });

        // Guidance slider
        this.elements.guidanceSlider?.addEventListener('input', (e) => {
            this.config.guidanceScale = parseFloat(e.target.value);
            this.elements.guidanceValue.textContent = this.config.guidanceScale.toFixed(1);
        });

        // Class select
        this.elements.classSelect?.addEventListener('change', (e) => {
            this.config.targetClass = parseInt(e.target.value);
        });

        // Backend select
        this.elements.backendSelect?.addEventListener('change', (e) => {
            this.config.backend = e.target.value;
            this.checkBackendSupport();
        });

        // Training mode toggle
        this.elements.trainModeToggle?.addEventListener('change', (e) => {
            const show = e.target.checked;
            this.elements.trainControls.style.display = show ? 'flex' : 'none';
        });

        // Grid canvas click
        this.elements.gridCanvas?.addEventListener('click', (e) => this.onGridClick(e));

        // Time scrubber
        this.elements.timeScrubber?.addEventListener('input', (e) => this.onScrubberChange(e));
    }

    /**
     * Check and set backend support
     */
    async checkBackendSupport() {
        const status = this.elements.backendStatus;

        if (this.config.backend === 'webgpu') {
            if (navigator.gpu) {
                try {
                    const adapter = await navigator.gpu.requestAdapter();
                    if (adapter) {
                        status?.classList.add('active');
                        this.updateStatus('WebGPU Ready');
                        return true;
                    }
                } catch (e) {
                    console.warn('[Diffusion] WebGPU not available:', e);
                }
            }
            status?.classList.remove('active');
            this.updateStatus('WebGPU Unavailable');
            return false;
        }

        // WebGL/CPU always available
        status?.classList.add('active');
        this.updateStatus('Ready');
        return true;
    }

    /**
     * Update status display
     */
    updateStatus(text) {
        if (this.elements.statusText) {
            this.elements.statusText.textContent = text;
        }
    }

    /**
     * Update progress bar
     */
    updateProgress(percent, text) {
        if (this.elements.progressBar) {
            this.elements.progressBar.style.width = `${percent}%`;
        }
        if (this.elements.progressText) {
            this.elements.progressText.textContent = text;
        }
    }

    /**
     * Generate digits
     */
    async generate() {
        if (this.isGenerating) return;

        this.isGenerating = true;
        this.updateStatus('Generating...');
        this.updateProgress(0, 'Starting...');

        // TODO: Implement actual generation in Phase 2
        // For now, simulate progress
        for (let i = 0; i <= 100; i += 5) {
            await new Promise(r => setTimeout(r, 50));
            this.updateProgress(i, `Step ${Math.floor(i / 5)} / 20`);
        }

        this.updateStatus('Ready');
        this.updateProgress(100, 'Complete');
        this.isGenerating = false;
    }

    /**
     * Handle grid canvas click
     */
    onGridClick(e) {
        const canvas = this.elements.gridCanvas;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Calculate which cell was clicked (4x4 grid)
        const cellSize = canvas.width / this.config.gridSize;
        const col = Math.floor(x / cellSize * (canvas.width / rect.width));
        const row = Math.floor(y / cellSize * (canvas.height / rect.height));
        const cellIndex = row * this.config.gridSize + col;

        console.log(`[Diffusion] Clicked cell ${cellIndex} (row=${row}, col=${col})`);
        this.selectCell(cellIndex);
    }

    /**
     * Select a cell to show in timeline
     */
    selectCell(index) {
        this.selectedCell = index;
        this.elements.previewHint.style.display = 'none';
        this.elements.scrubber.style.display = 'flex';

        // TODO: Show denoising timeline for this cell
        console.log(`[Diffusion] Selected cell ${index} for timeline view`);
    }

    /**
     * Handle timeline scrubber change
     */
    onScrubberChange(e) {
        const value = parseInt(e.target.value);
        const step = Math.floor(value / 100 * this.config.steps);
        this.elements.timeLabel.textContent = `t=${this.config.steps - step}`;

        // TODO: Update timeline canvas to show this step
    }

    /**
     * Open explainer modal
     */
    openExplainer() {
        // TODO: Implement in Phase 4
        console.log('[Diffusion] Open explainer');
    }

    /**
     * Cleanup when leaving tab
     */
    cleanup() {
        // Stop any ongoing generation
        this.isGenerating = false;
    }
}

// Export singleton instance
export const diffusionApp = new DiffusionApp();
```

**Step 2: Create diffusion-model.js skeleton**

Create `js/diffusion/diffusion-model.js`:

```javascript
/**
 * Diffusion Model - Tiny U-Net for MNIST generation
 * Implements DDPM/DDIM sampling with WebGPU acceleration
 */

export class DiffusionModel {
    constructor() {
        this.isLoaded = false;
        this.backend = 'webgpu';
        this.device = null;

        // Model config
        this.config = {
            imageSize: 28,
            channels: 1,
            numTimesteps: 1000,
            numClasses: 10
        };

        // Noise schedule
        this.betas = null;
        this.alphas = null;
        this.alphasCumprod = null;
    }

    /**
     * Initialize the model with specified backend
     */
    async init(backend = 'webgpu') {
        this.backend = backend;

        // Initialize noise schedule
        this.initNoiseSchedule();

        if (backend === 'webgpu') {
            await this.initWebGPU();
        }

        console.log(`[DiffusionModel] Initialized with ${backend} backend`);
    }

    /**
     * Initialize linear noise schedule
     */
    initNoiseSchedule() {
        const T = this.config.numTimesteps;
        const betaStart = 0.0001;
        const betaEnd = 0.02;

        // Linear schedule
        this.betas = new Float32Array(T);
        this.alphas = new Float32Array(T);
        this.alphasCumprod = new Float32Array(T);

        for (let t = 0; t < T; t++) {
            this.betas[t] = betaStart + (betaEnd - betaStart) * t / (T - 1);
            this.alphas[t] = 1 - this.betas[t];
            this.alphasCumprod[t] = t === 0
                ? this.alphas[t]
                : this.alphasCumprod[t - 1] * this.alphas[t];
        }
    }

    /**
     * Initialize WebGPU device and pipelines
     */
    async initWebGPU() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No WebGPU adapter found');
        }

        this.device = await adapter.requestDevice();
        console.log('[DiffusionModel] WebGPU device initialized');

        // TODO: Create compute pipelines in Phase 2
    }

    /**
     * Load pre-trained weights
     */
    async loadWeights(url) {
        // TODO: Implement weight loading
        console.log('[DiffusionModel] Loading weights from', url);
        this.isLoaded = true;
    }

    /**
     * Generate samples using DDPM
     * @param {number} numSamples - Number of samples to generate
     * @param {number} numSteps - Number of denoising steps
     * @param {number} classLabel - Class label (-1 for unconditional)
     * @param {number} guidanceScale - Classifier-free guidance scale
     * @param {Function} onProgress - Progress callback
     * @returns {Float32Array[]} Array of generated images
     */
    async sample(numSamples, numSteps, classLabel = -1, guidanceScale = 1.0, onProgress = null) {
        const { imageSize, channels } = this.config;
        const imagePixels = imageSize * imageSize * channels;

        // Start from pure noise
        const samples = new Array(numSamples);
        for (let i = 0; i < numSamples; i++) {
            samples[i] = this.randomNormal(imagePixels);
        }

        // Denoising loop
        const stepSize = Math.floor(this.config.numTimesteps / numSteps);

        for (let step = 0; step < numSteps; step++) {
            const t = this.config.numTimesteps - 1 - step * stepSize;

            // TODO: Implement actual denoising step
            // For now, just gradually reduce noise magnitude
            const noiseScale = Math.sqrt(this.alphasCumprod[t]);
            for (let i = 0; i < numSamples; i++) {
                for (let j = 0; j < imagePixels; j++) {
                    samples[i][j] *= 0.95;
                }
            }

            if (onProgress) {
                onProgress(step + 1, numSteps);
            }
        }

        return samples;
    }

    /**
     * Generate random normal values
     */
    randomNormal(size) {
        const arr = new Float32Array(size);
        for (let i = 0; i < size; i += 2) {
            const u1 = Math.random();
            const u2 = Math.random();
            const r = Math.sqrt(-2 * Math.log(u1));
            const theta = 2 * Math.PI * u2;
            arr[i] = r * Math.cos(theta);
            if (i + 1 < size) {
                arr[i + 1] = r * Math.sin(theta);
            }
        }
        return arr;
    }

    /**
     * Add noise to image at timestep t (forward diffusion)
     */
    addNoise(image, t) {
        const alphaCumprod = this.alphasCumprod[t];
        const noise = this.randomNormal(image.length);
        const result = new Float32Array(image.length);

        const sqrtAlpha = Math.sqrt(alphaCumprod);
        const sqrtOneMinusAlpha = Math.sqrt(1 - alphaCumprod);

        for (let i = 0; i < image.length; i++) {
            result[i] = sqrtAlpha * image[i] + sqrtOneMinusAlpha * noise[i];
        }

        return { noisy: result, noise };
    }
}

// Export singleton
export const diffusionModel = new DiffusionModel();
```

**Step 3: Create diffusion-viz.js skeleton**

Create `js/diffusion/diffusion-viz.js`:

```javascript
/**
 * Diffusion Visualization
 * Renders generation grid, denoising timeline, and U-Net architecture
 */

export class DiffusionViz {
    constructor() {
        this.gridCanvas = null;
        this.gridCtx = null;
        this.timelineCanvas = null;
        this.timelineCtx = null;
        this.archCanvas = null;
        this.archCtx = null;

        this.gridSize = 4;
        this.imageSize = 28;
        this.cellPadding = 8;
    }

    /**
     * Initialize visualization canvases
     */
    init(gridCanvas, timelineCanvas, archCanvas) {
        this.gridCanvas = gridCanvas;
        this.gridCtx = gridCanvas?.getContext('2d');

        this.timelineCanvas = timelineCanvas;
        this.timelineCtx = timelineCanvas?.getContext('2d');

        this.archCanvas = archCanvas;
        this.archCtx = archCanvas?.getContext('2d');

        // Draw initial states
        this.drawEmptyGrid();
        this.drawArchitecture();
    }

    /**
     * Draw empty placeholder grid
     */
    drawEmptyGrid() {
        if (!this.gridCtx) return;

        const ctx = this.gridCtx;
        const canvas = this.gridCanvas;
        const cellSize = canvas.width / this.gridSize;

        // Clear
        ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw cell borders
        ctx.strokeStyle = 'rgba(236, 72, 153, 0.3)';
        ctx.lineWidth = 1;

        for (let row = 0; row < this.gridSize; row++) {
            for (let col = 0; col < this.gridSize; col++) {
                const x = col * cellSize;
                const y = row * cellSize;

                ctx.strokeRect(
                    x + this.cellPadding / 2,
                    y + this.cellPadding / 2,
                    cellSize - this.cellPadding,
                    cellSize - this.cellPadding
                );

                // Draw placeholder "?"
                ctx.fillStyle = 'rgba(236, 72, 153, 0.2)';
                ctx.font = '24px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('?', x + cellSize / 2, y + cellSize / 2);
            }
        }
    }

    /**
     * Draw generated images to grid
     * @param {Float32Array[]} images - Array of 28x28 images
     */
    drawGrid(images) {
        if (!this.gridCtx) return;

        const ctx = this.gridCtx;
        const canvas = this.gridCanvas;
        const cellSize = canvas.width / this.gridSize;
        const imgSize = cellSize - this.cellPadding;

        // Clear
        ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw each image
        images.forEach((img, idx) => {
            const row = Math.floor(idx / this.gridSize);
            const col = idx % this.gridSize;
            const x = col * cellSize + this.cellPadding / 2;
            const y = row * cellSize + this.cellPadding / 2;

            this.drawImage(ctx, img, x, y, imgSize);
        });
    }

    /**
     * Draw a single image to canvas
     */
    drawImage(ctx, imageData, x, y, size) {
        const imgData = ctx.createImageData(this.imageSize, this.imageSize);

        for (let i = 0; i < imageData.length; i++) {
            // Normalize from [-1, 1] to [0, 255]
            const val = Math.floor((imageData[i] + 1) / 2 * 255);
            const clamped = Math.max(0, Math.min(255, val));

            imgData.data[i * 4] = clamped;     // R
            imgData.data[i * 4 + 1] = clamped; // G
            imgData.data[i * 4 + 2] = clamped; // B
            imgData.data[i * 4 + 3] = 255;     // A
        }

        // Create temp canvas for scaling
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.imageSize;
        tempCanvas.height = this.imageSize;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(imgData, 0, 0);

        // Draw scaled
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, x, y, size, size);
    }

    /**
     * Draw denoising timeline for a single image
     * @param {Float32Array[]} steps - Array of images at each timestep
     */
    drawTimeline(steps) {
        if (!this.timelineCtx) return;

        const ctx = this.timelineCtx;
        const canvas = this.timelineCanvas;

        // Clear
        ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        if (!steps || steps.length === 0) return;

        // Calculate layout
        const thumbSize = 36;
        const spacing = 8;
        const numSteps = Math.min(steps.length, 10); // Show max 10 steps
        const stepInterval = Math.floor(steps.length / numSteps);

        // Draw steps vertically
        for (let i = 0; i < numSteps; i++) {
            const stepIdx = i * stepInterval;
            const y = i * (thumbSize + spacing) + spacing;
            const x = (canvas.width - thumbSize) / 2;

            this.drawImage(ctx, steps[stepIdx], x, y, thumbSize);

            // Draw step label
            ctx.fillStyle = 'rgba(236, 72, 153, 0.8)';
            ctx.font = '10px monospace';
            ctx.textAlign = 'left';
            ctx.fillText(`t=${steps.length - stepIdx}`, x + thumbSize + 4, y + thumbSize / 2 + 3);
        }
    }

    /**
     * Draw mini U-Net architecture diagram
     */
    drawArchitecture() {
        if (!this.archCtx) return;

        const ctx = this.archCtx;
        const canvas = this.archCanvas;

        // Clear
        ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const centerX = canvas.width / 2;
        const boxWidth = 60;
        const boxHeight = 24;
        const spacing = 40;

        // Colors
        const encoderColor = '#6366f1';
        const decoderColor = '#8b5cf6';
        const skipColor = '#10b981';

        // Draw encoder path (left side going down)
        const encoderBlocks = ['Input', 'Down 1', 'Down 2', 'Bottleneck'];
        encoderBlocks.forEach((label, i) => {
            const x = centerX - boxWidth - 20;
            const y = 20 + i * spacing;

            this.drawBlock(ctx, x, y, boxWidth, boxHeight, label, encoderColor);

            // Draw down arrow
            if (i < encoderBlocks.length - 1) {
                this.drawArrow(ctx, x + boxWidth / 2, y + boxHeight, x + boxWidth / 2, y + spacing - 4, encoderColor);
            }
        });

        // Draw decoder path (right side going up)
        const decoderBlocks = ['Up 2', 'Up 1', 'Output'];
        decoderBlocks.forEach((label, i) => {
            const x = centerX + 20;
            const y = 20 + (3 - i) * spacing;

            this.drawBlock(ctx, x, y, boxWidth, boxHeight, label, decoderColor);

            // Draw up arrow
            if (i < decoderBlocks.length - 1) {
                this.drawArrow(ctx, x + boxWidth / 2, y, x + boxWidth / 2, y - spacing + boxHeight + 4, decoderColor);
            }
        });

        // Draw skip connections
        ctx.strokeStyle = skipColor;
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.5;

        // Skip from Down 1 to Up 1
        ctx.beginPath();
        ctx.moveTo(centerX - 20, 20 + spacing + boxHeight / 2);
        ctx.lineTo(centerX + 20, 20 + 2 * spacing + boxHeight / 2);
        ctx.stroke();

        // Skip from Down 2 to Up 2
        ctx.beginPath();
        ctx.moveTo(centerX - 20, 20 + 2 * spacing + boxHeight / 2);
        ctx.lineTo(centerX + 20, 20 + spacing + boxHeight / 2);
        ctx.stroke();

        ctx.setLineDash([]);

        // Draw time embedding injection
        ctx.fillStyle = 'rgba(236, 72, 153, 0.8)';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('+ time emb', centerX, canvas.height - 20);
        ctx.fillText('+ class emb', centerX, canvas.height - 8);
    }

    /**
     * Draw a block with label
     */
    drawBlock(ctx, x, y, width, height, label, color) {
        // Background
        ctx.fillStyle = color + '30';
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;

        ctx.beginPath();
        ctx.roundRect(x, y, width, height, 4);
        ctx.fill();
        ctx.stroke();

        // Label
        ctx.fillStyle = '#f8fafc';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, x + width / 2, y + height / 2);
    }

    /**
     * Draw an arrow
     */
    drawArrow(ctx, x1, y1, x2, y2, color) {
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1.5;

        // Line
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        // Arrowhead
        const angle = Math.atan2(y2 - y1, x2 - x1);
        const headLen = 6;

        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6), y2 - headLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6), y2 - headLen * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
    }

    /**
     * Highlight a cell in the grid
     */
    highlightCell(cellIndex) {
        if (!this.gridCtx) return;

        const ctx = this.gridCtx;
        const cellSize = this.gridCanvas.width / this.gridSize;
        const row = Math.floor(cellIndex / this.gridSize);
        const col = cellIndex % this.gridSize;
        const x = col * cellSize;
        const y = row * cellSize;

        ctx.strokeStyle = '#ec4899';
        ctx.lineWidth = 3;
        ctx.strokeRect(
            x + this.cellPadding / 2,
            y + this.cellPadding / 2,
            cellSize - this.cellPadding,
            cellSize - this.cellPadding
        );
    }
}

// Export singleton
export const diffusionViz = new DiffusionViz();
```

**Step 4: Update main.js to import diffusion module**

Read current main.js first, then add import and initialization:

Add near the top imports:
```javascript
import { diffusionApp } from './diffusion/diffusion-app.js';
import { diffusionViz } from './diffusion/diffusion-viz.js';
```

Add in the tab switching logic (find where other tabs are initialized):
```javascript
// In tab switching handler, add case for diffusion:
case 'diffusion':
    if (!diffusionApp.isInitialized) {
        diffusionApp.init();
        diffusionViz.init(
            document.getElementById('diffusion-grid-canvas'),
            document.getElementById('diffusion-timeline-canvas'),
            document.getElementById('diffusion-arch-mini-canvas')
        );
    }
    break;
```

**Step 5: Verify in browser**

1. Open browser with DevTools console
2. Click Diffusion tab
3. Should see "[Diffusion] Initializing..." and "[Diffusion] Initialized" in console
4. Click Generate button - should see progress bar animate
5. Click on grid - should see cell index logged

**Step 6: Commit**

```bash
git add js/diffusion/
git add js/main.js
git commit -m "feat: add diffusion tab JavaScript modules

- DiffusionApp: main orchestration and UI handlers
- DiffusionModel: U-Net skeleton with noise schedule
- DiffusionViz: canvas rendering for grid and timeline"
```

---

## Phase 2: U-Net Model Implementation

### Task 2.1: Implement Time Embedding

**Files:**
- Modify: `js/diffusion/diffusion-model.js`

**Step 1: Add sinusoidal time embedding**

Add this method to DiffusionModel class:

```javascript
/**
 * Create sinusoidal time embedding
 * @param {number} timestep - Current timestep
 * @param {number} dim - Embedding dimension
 * @returns {Float32Array} Time embedding vector
 */
getTimeEmbedding(timestep, dim = 64) {
    const halfDim = dim / 2;
    const embedding = new Float32Array(dim);

    // Sinusoidal embedding formula
    const logTimescale = Math.log(10000) / (halfDim - 1);

    for (let i = 0; i < halfDim; i++) {
        const freq = Math.exp(-i * logTimescale);
        const angle = timestep * freq;
        embedding[i] = Math.sin(angle);
        embedding[i + halfDim] = Math.cos(angle);
    }

    return embedding;
}
```

**Step 2: Test time embedding in console**

```javascript
// In browser console after loading diffusion tab:
const model = window.diffusionModel; // Need to expose for testing
const emb0 = model.getTimeEmbedding(0, 64);
const emb500 = model.getTimeEmbedding(500, 64);
const emb999 = model.getTimeEmbedding(999, 64);
console.log('t=0:', emb0.slice(0, 8));
console.log('t=500:', emb500.slice(0, 8));
console.log('t=999:', emb999.slice(0, 8));
// Should see different values for each timestep
```

**Step 3: Commit**

```bash
git add js/diffusion/diffusion-model.js
git commit -m "feat: add sinusoidal time embedding for U-Net"
```

---

### Task 2.2: Implement Basic Tensor Operations

**Files:**
- Create: `js/diffusion/tensor-ops.js`

**Step 1: Create tensor operations module**

```javascript
/**
 * Tensor Operations for Diffusion Model
 * Pure JavaScript implementations for CPU fallback
 */

export class TensorOps {
    /**
     * 2D Convolution
     * @param {Float32Array} input - [H, W, C_in] flattened
     * @param {Float32Array} kernel - [kH, kW, C_in, C_out] flattened
     * @param {number[]} inputShape - [H, W, C_in]
     * @param {number[]} kernelShape - [kH, kW, C_in, C_out]
     * @param {number} stride - Stride (default 1)
     * @param {string} padding - 'same' or 'valid'
     * @returns {Float32Array} Output tensor
     */
    static conv2d(input, kernel, inputShape, kernelShape, stride = 1, padding = 'same') {
        const [H, W, Cin] = inputShape;
        const [kH, kW, kCin, Cout] = kernelShape;

        // Calculate output dimensions
        let outH, outW, padH, padW;
        if (padding === 'same') {
            outH = Math.ceil(H / stride);
            outW = Math.ceil(W / stride);
            padH = Math.floor((kH - 1) / 2);
            padW = Math.floor((kW - 1) / 2);
        } else {
            outH = Math.floor((H - kH) / stride) + 1;
            outW = Math.floor((W - kW) / stride) + 1;
            padH = 0;
            padW = 0;
        }

        const output = new Float32Array(outH * outW * Cout);

        // Convolution
        for (let oh = 0; oh < outH; oh++) {
            for (let ow = 0; ow < outW; ow++) {
                for (let oc = 0; oc < Cout; oc++) {
                    let sum = 0;

                    for (let kh = 0; kh < kH; kh++) {
                        for (let kw = 0; kw < kW; kw++) {
                            const ih = oh * stride + kh - padH;
                            const iw = ow * stride + kw - padW;

                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                for (let ic = 0; ic < Cin; ic++) {
                                    const inputIdx = (ih * W + iw) * Cin + ic;
                                    const kernelIdx = ((kh * kW + kw) * kCin + ic) * Cout + oc;
                                    sum += input[inputIdx] * kernel[kernelIdx];
                                }
                            }
                        }
                    }

                    output[(oh * outW + ow) * Cout + oc] = sum;
                }
            }
        }

        return output;
    }

    /**
     * Group Normalization
     * @param {Float32Array} input - [H, W, C] flattened
     * @param {number[]} shape - [H, W, C]
     * @param {number} numGroups - Number of groups
     * @param {Float32Array} gamma - Scale parameter [C]
     * @param {Float32Array} beta - Shift parameter [C]
     * @param {number} eps - Epsilon for numerical stability
     */
    static groupNorm(input, shape, numGroups, gamma, beta, eps = 1e-5) {
        const [H, W, C] = shape;
        const groupSize = C / numGroups;
        const output = new Float32Array(input.length);

        for (let g = 0; g < numGroups; g++) {
            // Calculate mean and variance for this group
            let sum = 0;
            let sumSq = 0;
            let count = 0;

            for (let h = 0; h < H; h++) {
                for (let w = 0; w < W; w++) {
                    for (let c = g * groupSize; c < (g + 1) * groupSize; c++) {
                        const idx = (h * W + w) * C + c;
                        sum += input[idx];
                        sumSq += input[idx] * input[idx];
                        count++;
                    }
                }
            }

            const mean = sum / count;
            const variance = sumSq / count - mean * mean;
            const std = Math.sqrt(variance + eps);

            // Normalize
            for (let h = 0; h < H; h++) {
                for (let w = 0; w < W; w++) {
                    for (let c = g * groupSize; c < (g + 1) * groupSize; c++) {
                        const idx = (h * W + w) * C + c;
                        output[idx] = gamma[c] * (input[idx] - mean) / std + beta[c];
                    }
                }
            }
        }

        return output;
    }

    /**
     * SiLU (Swish) activation: x * sigmoid(x)
     */
    static silu(input) {
        const output = new Float32Array(input.length);
        for (let i = 0; i < input.length; i++) {
            const sigmoid = 1 / (1 + Math.exp(-input[i]));
            output[i] = input[i] * sigmoid;
        }
        return output;
    }

    /**
     * Bilinear upsample 2x
     */
    static upsample2x(input, shape) {
        const [H, W, C] = shape;
        const newH = H * 2;
        const newW = W * 2;
        const output = new Float32Array(newH * newW * C);

        for (let h = 0; h < newH; h++) {
            for (let w = 0; w < newW; w++) {
                const srcH = h / 2;
                const srcW = w / 2;

                const h0 = Math.floor(srcH);
                const h1 = Math.min(h0 + 1, H - 1);
                const w0 = Math.floor(srcW);
                const w1 = Math.min(w0 + 1, W - 1);

                const hLerp = srcH - h0;
                const wLerp = srcW - w0;

                for (let c = 0; c < C; c++) {
                    const v00 = input[(h0 * W + w0) * C + c];
                    const v01 = input[(h0 * W + w1) * C + c];
                    const v10 = input[(h1 * W + w0) * C + c];
                    const v11 = input[(h1 * W + w1) * C + c];

                    const v0 = v00 * (1 - wLerp) + v01 * wLerp;
                    const v1 = v10 * (1 - wLerp) + v11 * wLerp;

                    output[(h * newW + w) * C + c] = v0 * (1 - hLerp) + v1 * hLerp;
                }
            }
        }

        return output;
    }

    /**
     * Average pooling 2x downsample
     */
    static avgPool2x(input, shape) {
        const [H, W, C] = shape;
        const newH = Math.floor(H / 2);
        const newW = Math.floor(W / 2);
        const output = new Float32Array(newH * newW * C);

        for (let h = 0; h < newH; h++) {
            for (let w = 0; w < newW; w++) {
                for (let c = 0; c < C; c++) {
                    const v00 = input[((h * 2) * W + (w * 2)) * C + c];
                    const v01 = input[((h * 2) * W + (w * 2 + 1)) * C + c];
                    const v10 = input[((h * 2 + 1) * W + (w * 2)) * C + c];
                    const v11 = input[((h * 2 + 1) * W + (w * 2 + 1)) * C + c];

                    output[(h * newW + w) * C + c] = (v00 + v01 + v10 + v11) / 4;
                }
            }
        }

        return output;
    }

    /**
     * Add two tensors element-wise
     */
    static add(a, b) {
        const output = new Float32Array(a.length);
        for (let i = 0; i < a.length; i++) {
            output[i] = a[i] + b[i];
        }
        return output;
    }

    /**
     * Concatenate tensors along channel dimension
     */
    static concat(a, b, shape) {
        const [H, W, Ca] = shape;
        const Cb = b.length / (H * W);
        const Cout = Ca + Cb;
        const output = new Float32Array(H * W * Cout);

        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                for (let c = 0; c < Ca; c++) {
                    output[(h * W + w) * Cout + c] = a[(h * W + w) * Ca + c];
                }
                for (let c = 0; c < Cb; c++) {
                    output[(h * W + w) * Cout + Ca + c] = b[(h * W + w) * Cb + c];
                }
            }
        }

        return output;
    }
}
```

**Step 2: Verify operations work**

```javascript
// Test in console
import { TensorOps } from './js/diffusion/tensor-ops.js';

// Test SiLU
const x = new Float32Array([-2, -1, 0, 1, 2]);
console.log('SiLU:', TensorOps.silu(x));
// Expected: approximately [-0.27, -0.27, 0, 0.73, 1.76]
```

**Step 3: Commit**

```bash
git add js/diffusion/tensor-ops.js
git commit -m "feat: add tensor operations for U-Net (conv2d, groupnorm, etc.)"
```

---

### Task 2.3: Implement U-Net Architecture (CPU Version)

**Files:**
- Create: `js/diffusion/unet.js`

**Step 1: Create U-Net module**

```javascript
/**
 * Tiny U-Net for MNIST Diffusion
 * ~100K parameters, designed for browser inference
 */

import { TensorOps } from './tensor-ops.js';

export class UNet {
    constructor() {
        this.weights = null;
        this.config = {
            imageSize: 28,
            inChannels: 1,
            outChannels: 1,
            baseChannels: 32,
            channelMult: [1, 2, 4], // 32, 64, 128
            numResBlocks: 1,
            timeEmbedDim: 64,
            numClasses: 10,
            classEmbedDim: 64
        };
    }

    /**
     * Load weights from binary file
     */
    async loadWeights(url) {
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();
        this.weights = this.parseWeights(buffer);
        console.log('[UNet] Weights loaded');
    }

    /**
     * Initialize random weights for training
     */
    initRandomWeights() {
        const cfg = this.config;
        this.weights = {};

        // Helper to create random weights
        const randn = (size, scale = 0.02) => {
            const arr = new Float32Array(size);
            for (let i = 0; i < size; i++) {
                // Box-Muller transform
                const u1 = Math.random();
                const u2 = Math.random();
                arr[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scale;
            }
            return arr;
        };

        const zeros = (size) => new Float32Array(size);
        const ones = (size) => {
            const arr = new Float32Array(size);
            arr.fill(1);
            return arr;
        };

        // Time embedding MLP
        this.weights.timeEmbed1 = randn(cfg.timeEmbedDim * cfg.timeEmbedDim * 4);
        this.weights.timeEmbed1Bias = zeros(cfg.timeEmbedDim * 4);
        this.weights.timeEmbed2 = randn(cfg.timeEmbedDim * 4 * cfg.timeEmbedDim);
        this.weights.timeEmbed2Bias = zeros(cfg.timeEmbedDim);

        // Class embedding
        this.weights.classEmbed = randn(cfg.numClasses * cfg.classEmbedDim);

        // Input conv
        this.weights.convIn = randn(3 * 3 * cfg.inChannels * cfg.baseChannels);
        this.weights.convInBias = zeros(cfg.baseChannels);

        // Encoder blocks (simplified - just conv + norm)
        const channels = cfg.channelMult.map(m => m * cfg.baseChannels);

        // Down block 1: 32 -> 64
        this.weights.down1Conv = randn(3 * 3 * channels[0] * channels[1]);
        this.weights.down1Bias = zeros(channels[1]);
        this.weights.down1Gamma = ones(channels[1]);
        this.weights.down1Beta = zeros(channels[1]);

        // Down block 2: 64 -> 128
        this.weights.down2Conv = randn(3 * 3 * channels[1] * channels[2]);
        this.weights.down2Bias = zeros(channels[2]);
        this.weights.down2Gamma = ones(channels[2]);
        this.weights.down2Beta = zeros(channels[2]);

        // Bottleneck
        this.weights.midConv = randn(3 * 3 * channels[2] * channels[2]);
        this.weights.midBias = zeros(channels[2]);
        this.weights.midGamma = ones(channels[2]);
        this.weights.midBeta = zeros(channels[2]);

        // Up block 1: 128+128 -> 64
        this.weights.up1Conv = randn(3 * 3 * (channels[2] * 2) * channels[1]);
        this.weights.up1Bias = zeros(channels[1]);
        this.weights.up1Gamma = ones(channels[1]);
        this.weights.up1Beta = zeros(channels[1]);

        // Up block 2: 64+64 -> 32
        this.weights.up2Conv = randn(3 * 3 * (channels[1] * 2) * channels[0]);
        this.weights.up2Bias = zeros(channels[0]);
        this.weights.up2Gamma = ones(channels[0]);
        this.weights.up2Beta = zeros(channels[0]);

        // Output conv
        this.weights.convOut = randn(3 * 3 * channels[0] * cfg.outChannels);
        this.weights.convOutBias = zeros(cfg.outChannels);

        console.log('[UNet] Random weights initialized');
    }

    /**
     * Parse weights from binary buffer
     */
    parseWeights(buffer) {
        // TODO: Implement actual weight parsing
        // For now, initialize random weights
        this.initRandomWeights();
        return this.weights;
    }

    /**
     * Forward pass
     * @param {Float32Array} x - Noisy image [28, 28, 1]
     * @param {Float32Array} timeEmb - Time embedding [64]
     * @param {number} classLabel - Class label (0-9) or -1 for unconditional
     * @returns {Float32Array} Predicted noise [28, 28, 1]
     */
    forward(x, timeEmb, classLabel = -1) {
        const cfg = this.config;
        const channels = cfg.channelMult.map(m => m * cfg.baseChannels);

        // Process time embedding through MLP
        let t = this.timeEmbedMLP(timeEmb);

        // Add class embedding if conditional
        if (classLabel >= 0 && classLabel < cfg.numClasses) {
            const classEmb = this.getClassEmbedding(classLabel);
            t = TensorOps.add(t, classEmb);
        }

        // Input conv: [28, 28, 1] -> [28, 28, 32]
        let h = this.conv2d(x, [28, 28, 1], this.weights.convIn, [3, 3, 1, channels[0]], this.weights.convInBias);
        h = TensorOps.silu(h);

        // Down 1: [28, 28, 32] -> [14, 14, 64]
        const skip1 = h;
        h = this.resBlock(h, [28, 28, channels[0]], t, 'down1');
        h = TensorOps.avgPool2x(h, [28, 28, channels[1]]);

        // Down 2: [14, 14, 64] -> [7, 7, 128]
        const skip2 = h;
        h = this.resBlock(h, [14, 14, channels[1]], t, 'down2');
        h = TensorOps.avgPool2x(h, [14, 14, channels[2]]);

        // Bottleneck: [7, 7, 128]
        h = this.resBlock(h, [7, 7, channels[2]], t, 'mid');

        // Up 1: [7, 7, 128] -> [14, 14, 64]
        h = TensorOps.upsample2x(h, [7, 7, channels[2]]);
        h = TensorOps.concat(h, skip2, [14, 14, channels[2]]);
        h = this.resBlock(h, [14, 14, channels[2] * 2], t, 'up1');

        // Up 2: [14, 14, 64] -> [28, 28, 32]
        h = TensorOps.upsample2x(h, [14, 14, channels[1]]);
        h = TensorOps.concat(h, skip1, [28, 28, channels[1]]);
        h = this.resBlock(h, [28, 28, channels[1] * 2], t, 'up2');

        // Output conv: [28, 28, 32] -> [28, 28, 1]
        h = this.conv2d(h, [28, 28, channels[0]], this.weights.convOut, [3, 3, channels[0], 1], this.weights.convOutBias);

        return h;
    }

    /**
     * Time embedding MLP
     */
    timeEmbedMLP(timeEmb) {
        const dim = this.config.timeEmbedDim;

        // Linear 1: [64] -> [256]
        let h = this.linear(timeEmb, this.weights.timeEmbed1, this.weights.timeEmbed1Bias, dim, dim * 4);
        h = TensorOps.silu(h);

        // Linear 2: [256] -> [64]
        h = this.linear(h, this.weights.timeEmbed2, this.weights.timeEmbed2Bias, dim * 4, dim);

        return h;
    }

    /**
     * Get class embedding
     */
    getClassEmbedding(classLabel) {
        const dim = this.config.classEmbedDim;
        const start = classLabel * dim;
        return this.weights.classEmbed.slice(start, start + dim);
    }

    /**
     * Residual block with time conditioning
     */
    resBlock(x, shape, timeEmb, prefix) {
        const [H, W, C] = shape;
        const outC = this.weights[`${prefix}Conv`].length / (9 * C);

        // GroupNorm + SiLU + Conv
        let h = TensorOps.groupNorm(x, shape, 8, this.weights[`${prefix}Gamma`], this.weights[`${prefix}Beta`]);
        h = TensorOps.silu(h);
        h = this.conv2d(h, shape, this.weights[`${prefix}Conv`], [3, 3, C, outC], this.weights[`${prefix}Bias`]);

        // Add time embedding (broadcast across spatial dims)
        // Simplified: just scale by time embedding magnitude
        const timeScale = timeEmb.reduce((a, b) => a + b, 0) / timeEmb.length;
        for (let i = 0; i < h.length; i++) {
            h[i] *= (1 + timeScale * 0.1);
        }

        return h;
    }

    /**
     * Conv2D wrapper
     */
    conv2d(input, inputShape, kernel, kernelShape, bias) {
        let output = TensorOps.conv2d(input, kernel, inputShape, kernelShape, 1, 'same');

        // Add bias
        const [H, W, C] = [inputShape[0], inputShape[1], kernelShape[3]];
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                for (let c = 0; c < C; c++) {
                    output[(h * W + w) * C + c] += bias[c];
                }
            }
        }

        return output;
    }

    /**
     * Linear layer
     */
    linear(input, weights, bias, inDim, outDim) {
        const output = new Float32Array(outDim);

        for (let o = 0; o < outDim; o++) {
            let sum = bias[o];
            for (let i = 0; i < inDim; i++) {
                sum += input[i] * weights[i * outDim + o];
            }
            output[o] = sum;
        }

        return output;
    }

    /**
     * Count total parameters
     */
    countParams() {
        if (!this.weights) return 0;

        let total = 0;
        for (const key in this.weights) {
            total += this.weights[key].length;
        }
        return total;
    }
}

export const unet = new UNet();
```

**Step 2: Verify parameter count**

```javascript
// Test in console
import { unet } from './js/diffusion/unet.js';
unet.initRandomWeights();
console.log('Total params:', unet.countParams());
// Should be approximately 100K
```

**Step 3: Commit**

```bash
git add js/diffusion/unet.js
git commit -m "feat: implement Tiny U-Net architecture for MNIST diffusion"
```

---

### Task 2.4: Connect U-Net to Sampling Loop

**Files:**
- Modify: `js/diffusion/diffusion-model.js`

**Step 1: Update DiffusionModel to use U-Net**

Add import at top:
```javascript
import { unet } from './unet.js';
```

Update the `sample` method:

```javascript
/**
 * Generate samples using DDPM
 */
async sample(numSamples, numSteps, classLabel = -1, guidanceScale = 1.0, onProgress = null) {
    const { imageSize, channels, numTimesteps } = this.config;
    const imagePixels = imageSize * imageSize * channels;

    // Ensure U-Net is initialized
    if (!unet.weights) {
        unet.initRandomWeights();
    }

    // Start from pure noise
    const samples = [];
    const histories = []; // Store intermediate steps for visualization

    for (let i = 0; i < numSamples; i++) {
        samples.push(this.randomNormal(imagePixels));
        histories.push([this.randomNormal(imagePixels)]); // Save initial noise
    }

    // Denoising loop
    const stepIndices = this.getStepIndices(numSteps);

    for (let stepIdx = 0; stepIdx < stepIndices.length; stepIdx++) {
        const t = stepIndices[stepIdx];
        const tNext = stepIdx < stepIndices.length - 1 ? stepIndices[stepIdx + 1] : 0;

        // Get time embedding
        const timeEmb = this.getTimeEmbedding(t);

        for (let i = 0; i < numSamples; i++) {
            // Predict noise
            let predictedNoise;

            if (guidanceScale > 1.0 && classLabel >= 0) {
                // Classifier-free guidance
                const condNoise = unet.forward(samples[i], timeEmb, classLabel);
                const uncondNoise = unet.forward(samples[i], timeEmb, -1);

                predictedNoise = new Float32Array(imagePixels);
                for (let j = 0; j < imagePixels; j++) {
                    predictedNoise[j] = uncondNoise[j] + guidanceScale * (condNoise[j] - uncondNoise[j]);
                }
            } else {
                predictedNoise = unet.forward(samples[i], timeEmb, classLabel);
            }

            // DDPM step
            samples[i] = this.ddpmStep(samples[i], predictedNoise, t, tNext);

            // Save to history (every few steps)
            if (stepIdx % Math.max(1, Math.floor(numSteps / 10)) === 0) {
                histories[i].push(new Float32Array(samples[i]));
            }
        }

        if (onProgress) {
            onProgress(stepIdx + 1, numSteps);
        }

        // Yield to UI
        await new Promise(r => setTimeout(r, 0));
    }

    // Save final result to histories
    for (let i = 0; i < numSamples; i++) {
        histories[i].push(new Float32Array(samples[i]));
    }

    return { samples, histories };
}

/**
 * Get step indices for sampling
 */
getStepIndices(numSteps) {
    const indices = [];
    const stepSize = Math.floor(this.config.numTimesteps / numSteps);

    for (let i = 0; i < numSteps; i++) {
        indices.push(this.config.numTimesteps - 1 - i * stepSize);
    }

    return indices;
}

/**
 * Single DDPM denoising step
 */
ddpmStep(x, predictedNoise, t, tNext) {
    const alpha = this.alphas[t];
    const alphaCumprod = this.alphasCumprod[t];
    const alphaCumprodPrev = tNext > 0 ? this.alphasCumprod[tNext] : 1.0;
    const beta = this.betas[t];

    // Predicted x0
    const sqrtAlphaCumprod = Math.sqrt(alphaCumprod);
    const sqrtOneMinusAlphaCumprod = Math.sqrt(1 - alphaCumprod);

    const output = new Float32Array(x.length);

    // Mean
    const coef1 = 1 / Math.sqrt(alpha);
    const coef2 = beta / sqrtOneMinusAlphaCumprod;

    for (let i = 0; i < x.length; i++) {
        output[i] = coef1 * (x[i] - coef2 * predictedNoise[i]);
    }

    // Add noise (except for last step)
    if (tNext > 0) {
        const sigma = Math.sqrt(beta);
        const noise = this.randomNormal(x.length);
        for (let i = 0; i < x.length; i++) {
            output[i] += sigma * noise[i];
        }
    }

    return output;
}
```

**Step 2: Update diffusion-app.js to use new sampling**

Update the `generate` method in `diffusion-app.js`:

```javascript
/**
 * Generate digits
 */
async generate() {
    if (this.isGenerating) return;

    this.isGenerating = true;
    this.updateStatus('Generating...');
    this.updateProgress(0, 'Starting...');

    try {
        // Import model
        const { diffusionModel } = await import('./diffusion-model.js');

        // Initialize if needed
        if (!diffusionModel.alphas) {
            await diffusionModel.init(this.config.backend);
        }

        // Generate
        const numSamples = this.config.gridSize * this.config.gridSize;
        const result = await diffusionModel.sample(
            numSamples,
            this.config.steps,
            this.config.targetClass,
            this.config.guidanceScale,
            (step, total) => {
                const percent = (step / total) * 100;
                this.updateProgress(percent, `Step ${step}/${total}`);
            }
        );

        // Store for timeline
        this.generationHistory = result.histories;

        // Visualize
        const { diffusionViz } = await import('./diffusion-viz.js');
        diffusionViz.drawGrid(result.samples);

        this.updateStatus('Ready');
        this.updateProgress(100, 'Complete');

    } catch (error) {
        console.error('[Diffusion] Generation error:', error);
        this.updateStatus('Error');
    }

    this.isGenerating = false;
}
```

**Step 3: Verify generation works**

1. Open browser, click Diffusion tab
2. Click Generate button
3. Should see progress bar advancing
4. Grid should show generated noise patterns (will be random until weights are trained)

**Step 4: Commit**

```bash
git add js/diffusion/diffusion-model.js js/diffusion/diffusion-app.js
git commit -m "feat: connect U-Net to DDPM sampling loop"
```

---

## Phase 3: Visualization Enhancement

### Task 3.1: Implement Grid Cell Selection and Timeline

**Files:**
- Modify: `js/diffusion/diffusion-app.js`
- Modify: `js/diffusion/diffusion-viz.js`

**Step 1: Update selectCell in diffusion-app.js**

```javascript
/**
 * Select a cell to show in timeline
 */
selectCell(index) {
    this.selectedCell = index;
    this.elements.previewHint.style.display = 'none';
    this.elements.scrubber.style.display = 'flex';

    // Draw timeline for selected cell
    if (this.generationHistory && this.generationHistory[index]) {
        const { diffusionViz } = await import('./diffusion-viz.js');
        diffusionViz.drawTimeline(this.generationHistory[index]);
        diffusionViz.highlightCell(index);
    }
}
```

**Step 2: Update drawGrid to support cell highlighting**

In `diffusion-viz.js`, update `drawGrid`:

```javascript
/**
 * Draw generated images to grid
 * @param {Float32Array[]} images - Array of 28x28 images
 * @param {number} selectedIndex - Index of selected cell (-1 for none)
 */
drawGrid(images, selectedIndex = -1) {
    if (!this.gridCtx) return;

    const ctx = this.gridCtx;
    const canvas = this.gridCanvas;
    const cellSize = canvas.width / this.gridSize;
    const imgSize = cellSize - this.cellPadding;

    // Clear
    ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw each image
    images.forEach((img, idx) => {
        const row = Math.floor(idx / this.gridSize);
        const col = idx % this.gridSize;
        const x = col * cellSize + this.cellPadding / 2;
        const y = row * cellSize + this.cellPadding / 2;

        this.drawImage(ctx, img, x, y, imgSize);

        // Highlight if selected
        if (idx === selectedIndex) {
            ctx.strokeStyle = '#ec4899';
            ctx.lineWidth = 3;
            ctx.strokeRect(x - 2, y - 2, imgSize + 4, imgSize + 4);
        }
    });
}
```

**Step 3: Test cell selection**

1. Generate some images
2. Click on a cell
3. Should see timeline update and cell highlighted

**Step 4: Commit**

```bash
git add js/diffusion/diffusion-app.js js/diffusion/diffusion-viz.js
git commit -m "feat: add cell selection and timeline visualization"
```

---

## Phase 4-6: Explainer, Training Mode, and Polish

_Phases 4-6 follow similar patterns. Key tasks include:_

### Phase 4 Tasks (Explainer Modal):
- 4.1: Create explainer modal HTML structure
- 4.2: Add explainer CSS styles
- 4.3: Implement Tab 1 (What is Diffusion) with animation
- 4.4: Implement Tab 2 (Forward Process) with interactive slider
- 4.5: Implement Tab 3 (Reverse Process) with step animation
- 4.6: Implement Tab 4 (U-Net Architecture) with interactive diagram
- 4.7: Implement Tab 5 (Noise Schedules) with comparison
- 4.8: Implement Tab 6 (Sampling Methods) with race visualization
- 4.9: Implement Tab 7 (Guidance) with scale slider

### Phase 5 Tasks (Training Mode):
- 5.1: Add TensorFlow.js integration for training
- 5.2: Implement training loop with loss calculation
- 5.3: Add loss visualization chart
- 5.4: Implement weight save/load to IndexedDB
- 5.5: Add sample generation during training

### Phase 6 Tasks (Polish):
- 6.1: Add loading animations and transitions
- 6.2: Implement educational tooltips
- 6.3: Add responsive breakpoints
- 6.4: Performance optimization (Web Workers)
- 6.5: Pre-trained weight loading from CDN
- 6.6: Final testing and bug fixes

---

## Testing Checklist

### Manual Browser Tests:
- [ ] Tab navigation works, initializes on first click
- [ ] WebGPU detection and fallback to WebGL/CPU
- [ ] Generate button produces visible output
- [ ] Progress bar updates during generation
- [ ] Grid cell click selects and shows timeline
- [ ] Timeline scrubber changes displayed step
- [ ] All 7 explainer tabs render correctly
- [ ] Training mode toggle shows/hides controls
- [ ] Training produces decreasing loss over epochs
- [ ] Responsive layout at 1200px, 900px, 600px widths

### Console Error Checks:
- [ ] No errors on initial page load
- [ ] No errors when switching to Diffusion tab
- [ ] No errors during generation
- [ ] No errors when opening explainer

---

## Appendix: File Reference

| File | Purpose |
|------|---------|
| `index.html` | Tab button + panel HTML |
| `css/diffusion.css` | All diffusion-specific styles |
| `js/diffusion/diffusion-app.js` | Main orchestration, UI events |
| `js/diffusion/diffusion-model.js` | DDPM/DDIM sampling, noise schedule |
| `js/diffusion/diffusion-viz.js` | Canvas rendering (grid, timeline, arch) |
| `js/diffusion/diffusion-explainer.js` | 7-tab explainer logic |
| `js/diffusion/unet.js` | Tiny U-Net architecture |
| `js/diffusion/tensor-ops.js` | CPU tensor operations |
| `js/main.js` | Import + tab switching integration |
