/**
 * CNN Dataset Loader
 * Supports MNIST, Fashion MNIST, and CIFAR-10 datasets
 */

// MNIST and Fashion MNIST share the same format
const DATASETS = {
    mnist: {
        name: 'MNIST Digits',
        imagesUrl: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png',
        labelsUrl: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8',
        imageSize: 28,
        channels: 1,
        numClasses: 10,
        numElements: 65000,
        numTrain: 55000,
        classNames: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    },
    fashion: {
        name: 'Fashion MNIST',
        imagesUrl: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_images.png',
        labelsUrl: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_labels_uint8',
        imageSize: 28,
        channels: 1,
        numClasses: 10,
        numElements: 65000,
        numTrain: 55000,
        classNames: ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    },
    cifar10: {
        name: 'CIFAR-10',
        // CIFAR-10 needs special handling - we'll use a subset
        imagesUrl: 'https://storage.googleapis.com/learnjs-data/cifar10/cifar10_images.png',
        labelsUrl: 'https://storage.googleapis.com/learnjs-data/cifar10/cifar10_labels_uint8',
        imageSize: 32,
        channels: 3,
        numClasses: 10,
        numElements: 60000,
        numTrain: 50000,
        classNames: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    }
};

export class DatasetLoader {
    constructor() {
        this.currentDataset = null;
        this.datasetConfig = null;
        this.datasetImages = null;
        this.datasetLabels = null;
        this.trainIndices = null;
        this.testIndices = null;
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
        this.isLoaded = false;
        this.loadingProgress = 0;
    }

    /**
     * Get available datasets
     */
    static getAvailableDatasets() {
        return Object.entries(DATASETS).map(([key, config]) => ({
            id: key,
            name: config.name,
            imageSize: config.imageSize,
            channels: config.channels,
            numClasses: config.numClasses,
            classNames: config.classNames
        }));
    }

    /**
     * Get current dataset config
     */
    getConfig() {
        return this.datasetConfig;
    }

    /**
     * Get input shape for model
     */
    getInputShape() {
        if (!this.datasetConfig) return [28, 28, 1];
        return [this.datasetConfig.imageSize, this.datasetConfig.imageSize, this.datasetConfig.channels];
    }

    /**
     * Load dataset from remote sources
     * @param {string} datasetId - 'mnist', 'fashion', or 'cifar10'
     * @param {function} onProgress - Progress callback (0-100)
     */
    async load(datasetId = 'mnist', onProgress = null) {
        const config = DATASETS[datasetId];
        if (!config) {
            throw new Error(`Unknown dataset: ${datasetId}`);
        }

        // Reset state
        this.isLoaded = false;
        this.currentDataset = datasetId;
        this.datasetConfig = config;
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;

        const imageSize = config.imageSize * config.imageSize * config.channels;

        // Load images as sprite sheet
        const imgRequest = new Promise((resolve, reject) => {
            const img = new Image();
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            img.crossOrigin = 'anonymous';
            img.onload = () => {
                img.width = img.naturalWidth;
                img.height = img.naturalHeight;

                const datasetBytesBuffer = new ArrayBuffer(config.numElements * imageSize * 4);
                const chunkSize = 5000;

                canvas.width = img.width;
                canvas.height = chunkSize;

                const numChunks = Math.ceil(config.numElements / chunkSize);

                for (let i = 0; i < numChunks; i++) {
                    const actualChunkSize = Math.min(chunkSize, config.numElements - i * chunkSize);
                    const datasetBytesView = new Float32Array(
                        datasetBytesBuffer,
                        i * imageSize * chunkSize * 4,
                        imageSize * actualChunkSize
                    );

                    ctx.drawImage(
                        img,
                        0, i * chunkSize,
                        img.width, actualChunkSize,
                        0, 0,
                        img.width, actualChunkSize
                    );

                    const imageData = ctx.getImageData(0, 0, canvas.width, actualChunkSize);

                    if (config.channels === 1) {
                        // Grayscale - use red channel
                        for (let j = 0; j < actualChunkSize * config.imageSize * config.imageSize; j++) {
                            datasetBytesView[j] = imageData.data[j * 4] / 255;
                        }
                    } else {
                        // RGB - use all 3 channels
                        let idx = 0;
                        for (let j = 0; j < imageData.data.length; j += 4) {
                            datasetBytesView[idx++] = imageData.data[j] / 255;     // R
                            datasetBytesView[idx++] = imageData.data[j + 1] / 255; // G
                            datasetBytesView[idx++] = imageData.data[j + 2] / 255; // B
                        }
                    }

                    if (onProgress) {
                        this.loadingProgress = Math.floor((i + 1) / numChunks * 50);
                        onProgress(this.loadingProgress);
                    }
                }

                this.datasetImages = new Float32Array(datasetBytesBuffer);
                resolve();
            };

            img.onerror = () => {
                // Fallback: generate synthetic data for demo
                console.warn(`Failed to load ${config.name} images, generating synthetic data`);
                this.generateSyntheticData(config);
                resolve();
            };

            img.src = config.imagesUrl;
        });

        // Load labels
        const labelsRequest = fetch(config.labelsUrl)
            .then(response => {
                if (!response.ok) throw new Error('Labels fetch failed');
                return response.arrayBuffer();
            })
            .then(buffer => {
                this.datasetLabels = new Uint8Array(buffer);
                if (onProgress) {
                    this.loadingProgress = 75;
                    onProgress(this.loadingProgress);
                }
            })
            .catch(() => {
                // Fallback: generate random labels
                console.warn(`Failed to load ${config.name} labels, generating synthetic`);
                this.datasetLabels = new Uint8Array(config.numElements);
                for (let i = 0; i < config.numElements; i++) {
                    this.datasetLabels[i] = Math.floor(Math.random() * config.numClasses);
                }
            });

        await Promise.all([imgRequest, labelsRequest]);

        // Create shuffled indices
        this.trainIndices = tf.util.createShuffledIndices(config.numTrain);
        this.testIndices = tf.util.createShuffledIndices(config.numElements - config.numTrain);

        this.isLoaded = true;
        if (onProgress) {
            this.loadingProgress = 100;
            onProgress(100);
        }

        console.log(`${config.name} loaded:`, config.numTrain, 'train,', config.numElements - config.numTrain, 'test images');
    }

    /**
     * Generate synthetic data for fallback
     */
    generateSyntheticData(config) {
        const imageSize = config.imageSize * config.imageSize * config.channels;
        this.datasetImages = new Float32Array(config.numElements * imageSize);

        // Generate simple patterns for each class
        for (let i = 0; i < config.numElements; i++) {
            const classId = i % config.numClasses;
            const offset = i * imageSize;

            // Create different patterns based on class
            for (let y = 0; y < config.imageSize; y++) {
                for (let x = 0; x < config.imageSize; x++) {
                    for (let c = 0; c < config.channels; c++) {
                        const idx = offset + (y * config.imageSize + x) * config.channels + c;
                        // Simple geometric patterns
                        const cx = config.imageSize / 2;
                        const cy = config.imageSize / 2;
                        const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
                        const angle = Math.atan2(y - cy, x - cx);

                        let value = 0;
                        switch (classId) {
                            case 0: value = dist < 8 ? 1 : 0; break;  // Circle
                            case 1: value = Math.abs(x - cx) < 3 ? 1 : 0; break;  // Vertical line
                            case 2: value = Math.abs(y - cy) < 3 ? 1 : 0; break;  // Horizontal line
                            case 3: value = Math.abs(x - y) < 3 ? 1 : 0; break;  // Diagonal
                            case 4: value = (x + y) % 4 < 2 ? 1 : 0; break;  // Checkerboard
                            case 5: value = x < cx ? 1 : 0; break;  // Half
                            case 6: value = dist < 10 && dist > 5 ? 1 : 0; break;  // Ring
                            case 7: value = (x < 8 || x > 20) && (y < 8 || y > 20) ? 1 : 0; break;  // Border
                            case 8: value = Math.sin(angle * 4) > 0 ? 1 : 0; break;  // Sectors
                            case 9: value = (x % 4 < 2) !== (y % 4 < 2) ? 1 : 0; break;  // Grid
                        }
                        this.datasetImages[idx] = value * 0.8 + Math.random() * 0.2;
                    }
                }
            }
        }
    }

    /**
     * Get next training batch
     * @param {number} batchSize
     * @returns {{xs: tf.Tensor, ys: tf.Tensor}}
     */
    nextTrainBatch(batchSize) {
        return this.nextBatch(batchSize, () => {
            this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length;
            return this.trainIndices[this.shuffledTrainIndex];
        });
    }

    /**
     * Get next test batch
     * @param {number} batchSize
     * @returns {{xs: tf.Tensor, ys: tf.Tensor}}
     */
    nextTestBatch(batchSize) {
        return this.nextBatch(batchSize, () => {
            this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length;
            return this.testIndices[this.shuffledTestIndex] + this.datasetConfig.numTrain;
        });
    }

    /**
     * Internal batch fetching
     */
    nextBatch(batchSize, indexFn) {
        const config = this.datasetConfig;
        const imageSize = config.imageSize * config.imageSize * config.channels;

        const batchImagesArray = new Float32Array(batchSize * imageSize);
        const batchLabelsArray = new Uint8Array(batchSize * config.numClasses);

        for (let i = 0; i < batchSize; i++) {
            const idx = indexFn();

            const image = this.datasetImages.slice(idx * imageSize, idx * imageSize + imageSize);
            batchImagesArray.set(image, i * imageSize);

            const label = this.datasetLabels[idx];
            batchLabelsArray[i * config.numClasses + label] = 1;
        }

        const xs = tf.tensor4d(
            batchImagesArray,
            [batchSize, config.imageSize, config.imageSize, config.channels]
        );
        const ys = tf.tensor2d(batchLabelsArray, [batchSize, config.numClasses]);

        return { xs, ys };
    }

    /**
     * Get a single random sample for visualization
     * @param {boolean} fromTest - Whether to sample from test set
     * @returns {{image: tf.Tensor, label: number, imageData: Float32Array}}
     */
    getRandomSample(fromTest = false) {
        const config = this.datasetConfig;
        const imageSize = config.imageSize * config.imageSize * config.channels;
        const data = fromTest ? this.testIndices : this.trainIndices;
        const offset = fromTest ? config.numTrain : 0;
        const idx = data[Math.floor(Math.random() * data.length)] + offset;

        const imageData = this.datasetImages.slice(idx * imageSize, idx * imageSize + imageSize);
        const image = tf.tensor4d(
            imageData,
            [1, config.imageSize, config.imageSize, config.channels]
        );
        const label = this.datasetLabels[idx];

        return { image, label, imageData };
    }

    /**
     * Render image data to canvas
     * @param {Float32Array} imageData - Flat image data
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} scale - Scale factor
     */
    renderToCanvas(imageData, ctx, x, y, scale = 1) {
        const config = this.datasetConfig;
        const size = config.imageSize;

        for (let py = 0; py < size; py++) {
            for (let px = 0; px < size; px++) {
                if (config.channels === 1) {
                    const val = Math.floor(imageData[py * size + px] * 255);
                    ctx.fillStyle = `rgb(${val}, ${val}, ${val})`;
                } else {
                    const idx = (py * size + px) * 3;
                    const r = Math.floor(imageData[idx] * 255);
                    const g = Math.floor(imageData[idx + 1] * 255);
                    const b = Math.floor(imageData[idx + 2] * 255);
                    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                }
                ctx.fillRect(x + px * scale, y + py * scale, scale, scale);
            }
        }
    }
}
