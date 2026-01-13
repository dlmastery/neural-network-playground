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
