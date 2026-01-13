/**
 * Diffusion Explainer Modal
 * Handles the 7-tab educational explainer for diffusion concepts
 */

export class DiffusionExplainer {
    constructor() {
        this.modal = null;
        this.activeTab = 'd-overview';
        this.animationFrame = null;
        this.isOpen = false;
    }

    /**
     * Initialize the explainer
     */
    init() {
        this.modal = document.getElementById('diffusion-explainer-modal');

        // Tab buttons
        const tabButtons = this.modal?.querySelectorAll('.explainer-tab-btn');
        tabButtons?.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabId = btn.dataset.tab;
                this.switchTab(tabId);
            });
        });

        // Close button
        const closeBtn = document.getElementById('diffusion-explainer-close-btn');
        closeBtn?.addEventListener('click', () => this.close());

        // Click outside to close
        this.modal?.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.close();
            }
        });

        // ESC key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.close();
            }
        });

        // Initialize tab-specific controls
        this.initForwardProcess();
        this.initReverseProcess();
    }

    /**
     * Initialize forward process tab controls
     */
    initForwardProcess() {
        const slider = document.getElementById('forward-noise-slider');
        const valueDisplay = document.getElementById('forward-noise-value');

        slider?.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            if (valueDisplay) {
                valueDisplay.textContent = `t=${value}`;
            }
            this.drawForwardProcess(value);
        });
    }

    /**
     * Initialize reverse process tab controls
     */
    initReverseProcess() {
        const playBtn = document.getElementById('reverse-play-btn');

        playBtn?.addEventListener('click', () => {
            this.animateReverseProcess();
        });
    }

    /**
     * Open the explainer modal
     */
    open() {
        if (!this.modal) this.init();

        this.modal?.classList.add('active');
        this.isOpen = true;

        // Start animation for active tab
        this.startTabAnimation(this.activeTab);
    }

    /**
     * Close the explainer modal
     */
    close() {
        this.modal?.classList.remove('active');
        this.isOpen = false;
        this.stopAnimation();
    }

    /**
     * Switch to a different tab
     */
    switchTab(tabId) {
        // Update active button
        const buttons = this.modal?.querySelectorAll('.explainer-tab-btn');
        buttons?.forEach(btn => {
            btn.classList.toggle('explainer-tab-btn--active', btn.dataset.tab === tabId);
        });

        // Update active panel
        const panels = this.modal?.querySelectorAll('.explainer-tab-panel');
        panels?.forEach(panel => {
            const panelId = tabId + '-panel';
            panel.classList.toggle('explainer-tab-panel--active', panel.id === panelId);
        });

        this.activeTab = tabId;
        this.stopAnimation();
        this.startTabAnimation(tabId);
    }

    /**
     * Start animation for the active tab
     */
    startTabAnimation(tabId) {
        switch (tabId) {
            case 'd-overview':
                this.drawOverviewDiagram();
                break;
            case 'd-forward':
                this.drawForwardProcess(0);
                break;
            case 'd-reverse':
                this.drawReverseProcess(0);
                break;
            // Other tabs will be added in subsequent tasks
        }
    }

    /**
     * Stop any running animation
     */
    stopAnimation() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }

    /**
     * Draw the forward process visualization
     * @param {number} currentStep - Current noise level (0-100)
     */
    drawForwardProcess(currentStep = 0) {
        const canvas = document.getElementById('forward-process-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear
        ctx.fillStyle = 'rgba(15, 23, 42, 1)';
        ctx.fillRect(0, 0, width, height);

        // Draw sequence of noised digits
        const numSteps = 10;
        const boxSize = 36;
        const spacing = (width - numSteps * boxSize) / (numSteps + 1);
        const y = 20;

        for (let i = 0; i < numSteps; i++) {
            const x = spacing + i * (boxSize + spacing);
            const stepNoise = i / (numSteps - 1); // 0 to 1
            const isHighlighted = Math.abs((currentStep / 100) - stepNoise) < 0.1;

            // Draw digit with noise
            this.drawNoisyDigit(ctx, x, y, boxSize, stepNoise, '7');

            // Highlight current step
            if (isHighlighted) {
                ctx.strokeStyle = '#ec4899';
                ctx.lineWidth = 3;
                ctx.strokeRect(x - 2, y - 2, boxSize + 4, boxSize + 4);
            }
        }

        // Draw beta schedule graph below
        const graphY = y + boxSize + 20;
        const graphHeight = height - graphY - 10;
        const graphWidth = width - 40;
        const graphX = 20;

        // Graph background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.fillRect(graphX, graphY, graphWidth, graphHeight);

        // Draw beta curve (linear schedule)
        ctx.strokeStyle = '#ec4899';
        ctx.lineWidth = 2;
        ctx.beginPath();

        const betaStart = 0.0001;
        const betaEnd = 0.02;

        for (let i = 0; i <= graphWidth; i++) {
            const t = i / graphWidth;
            const beta = betaStart + t * (betaEnd - betaStart);
            const normalizedBeta = (beta - betaStart) / (betaEnd - betaStart);
            const plotY = graphY + graphHeight - normalizedBeta * graphHeight;

            if (i === 0) {
                ctx.moveTo(graphX + i, plotY);
            } else {
                ctx.lineTo(graphX + i, plotY);
            }
        }
        ctx.stroke();

        // Draw current position marker
        const markerX = graphX + (currentStep / 100) * graphWidth;
        ctx.fillStyle = '#ec4899';
        ctx.beginPath();
        ctx.arc(markerX, graphY + graphHeight - (currentStep / 100) * graphHeight, 5, 0, Math.PI * 2);
        ctx.fill();

        // Labels
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('β_t', graphX + 5, graphY + 12);
        ctx.textAlign = 'center';
        ctx.fillText('0', graphX, graphY + graphHeight + 12);
        ctx.fillText('T', graphX + graphWidth, graphY + graphHeight + 12);
    }

    /**
     * Draw the reverse process visualization
     * @param {number} currentStep - Current denoising step (0 = noise, 100 = clean)
     */
    drawReverseProcess(currentStep = 0) {
        const canvas = document.getElementById('reverse-process-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear
        ctx.fillStyle = 'rgba(15, 23, 42, 1)';
        ctx.fillRect(0, 0, width, height);

        // Main digit in center
        const mainSize = 120;
        const mainX = (width - mainSize) / 2;
        const mainY = 20;
        const noiseLevel = 1 - (currentStep / 100);

        // Draw large noisy digit
        this.drawNoisyDigit(ctx, mainX, mainY, mainSize, noiseLevel, '5');

        // Draw progress text
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`Denoising Step: ${currentStep}/100`, width / 2, mainY + mainSize + 25);

        // Draw noise level indicator
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px sans-serif';
        ctx.fillText(`Noise Level: ${Math.round(noiseLevel * 100)}%`, width / 2, mainY + mainSize + 45);

        // Draw progress bar
        const barWidth = 200;
        const barHeight = 8;
        const barX = (width - barWidth) / 2;
        const barY = height - 30;

        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.fillRect(barX, barY, barWidth, barHeight);

        // Progress
        ctx.fillStyle = '#ec4899';
        ctx.fillRect(barX, barY, barWidth * (currentStep / 100), barHeight);

        // Border
        ctx.strokeStyle = 'rgba(236, 72, 153, 0.5)';
        ctx.lineWidth = 1;
        ctx.strokeRect(barX, barY, barWidth, barHeight);

        // Labels
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('t=T (noise)', barX, barY - 5);
        ctx.textAlign = 'right';
        ctx.fillText('t=0 (clean)', barX + barWidth, barY - 5);
    }

    /**
     * Animate the reverse process
     */
    animateReverseProcess() {
        let step = 0;
        const totalSteps = 100;
        const stepDuration = 30; // ms per step

        // Update button text
        const playBtn = document.getElementById('reverse-play-btn');
        if (playBtn) {
            playBtn.textContent = '⏸ Running...';
            playBtn.disabled = true;
        }

        const animate = () => {
            this.drawReverseProcess(step);
            step++;

            if (step <= totalSteps) {
                this.animationFrame = requestAnimationFrame(() => {
                    setTimeout(animate, stepDuration);
                });
            } else {
                // Animation complete
                if (playBtn) {
                    playBtn.textContent = '▶ Play Again';
                    playBtn.disabled = false;
                }
            }
        };

        // Start from noise
        this.stopAnimation();
        animate();
    }

    /**
     * Draw the overview diagram showing forward and reverse process
     */
    drawOverviewDiagram() {
        const canvas = document.getElementById('diffusion-overview-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        let time = 0;
        const totalSteps = 5;

        const animate = () => {
            ctx.fillStyle = 'rgba(15, 23, 42, 1)';
            ctx.fillRect(0, 0, width, height);

            const step = Math.floor((time % 300) / 60); // 0-4 steps, cycling
            const progress = (time % 60) / 60; // 0-1 within each step

            // Draw title
            ctx.fillStyle = '#f8fafc';
            ctx.font = 'bold 14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Forward Process', width * 0.25, 20);
            ctx.fillText('Reverse Process', width * 0.75, 20);

            // Draw arrows
            ctx.fillStyle = '#ec4899';
            ctx.font = '20px sans-serif';
            ctx.fillText('→', width * 0.25, height / 2);
            ctx.fillText('←', width * 0.75, height / 2);

            // Draw digit boxes for forward process (left side)
            const boxSize = 32;
            const startX = 20;
            const y = height / 2 - boxSize / 2 + 10;

            for (let i = 0; i <= totalSteps; i++) {
                const x = startX + i * (boxSize + 8);
                const noiseLevel = i / totalSteps;

                this.drawNoisyDigit(ctx, x, y, boxSize, noiseLevel, '3');
            }

            // Draw digit boxes for reverse process (right side)
            const rightStartX = width / 2 + 20;

            for (let i = 0; i <= totalSteps; i++) {
                const x = rightStartX + i * (boxSize + 8);
                const noiseLevel = 1 - (i / totalSteps); // Reverse: starts noisy, ends clean

                this.drawNoisyDigit(ctx, x, y, boxSize, noiseLevel, '3');
            }

            // Draw step indicators
            ctx.fillStyle = '#94a3b8';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('t=0', startX + boxSize/2, y + boxSize + 15);
            ctx.fillText('t=T', startX + totalSteps * (boxSize + 8) + boxSize/2, y + boxSize + 15);
            ctx.fillText('t=T', rightStartX + boxSize/2, y + boxSize + 15);
            ctx.fillText('t=0', rightStartX + totalSteps * (boxSize + 8) + boxSize/2, y + boxSize + 15);

            time++;
            this.animationFrame = requestAnimationFrame(animate);
        };

        animate();
    }

    /**
     * Draw a digit with noise overlay
     */
    drawNoisyDigit(ctx, x, y, size, noiseLevel, digit) {
        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.fillRect(x, y, size, size);

        // Draw simple digit shape (simplified "3")
        ctx.fillStyle = `rgba(248, 250, 252, ${1 - noiseLevel * 0.8})`;
        ctx.font = `bold ${size * 0.7}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(digit, x + size/2, y + size/2);

        // Add noise dots
        const numDots = Math.floor(noiseLevel * 50);
        for (let i = 0; i < numDots; i++) {
            const dotX = x + Math.random() * size;
            const dotY = y + Math.random() * size;
            const brightness = Math.random() * 255;
            ctx.fillStyle = `rgba(${brightness}, ${brightness}, ${brightness}, ${noiseLevel * 0.5})`;
            ctx.fillRect(dotX, dotY, 2, 2);
        }

        // Border
        ctx.strokeStyle = 'rgba(236, 72, 153, 0.5)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, size, size);
    }
}

// Export singleton
export const diffusionExplainer = new DiffusionExplainer();
