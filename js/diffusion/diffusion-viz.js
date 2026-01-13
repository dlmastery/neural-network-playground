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
