/**
 * MNIST Dataset Loader
 * Loads and caches MNIST handwritten digit data for training CNNs
 */

const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

const IMAGE_SIZE = 784;  // 28 * 28
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

export class MNISTLoader {
    constructor() {
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
     * Load MNIST dataset from remote sources
     * @param {function} onProgress - Progress callback (0-100)
     */
    async load(onProgress = null) {
        if (this.isLoaded) return;

        // Load images as sprite sheet
        const imgRequest = new Promise((resolve, reject) => {
            const img = new Image();
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            img.crossOrigin = '';
            img.onload = () => {
                img.width = img.naturalWidth;
                img.height = img.naturalHeight;

                const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
                const chunkSize = 5000;

                canvas.width = img.width;
                canvas.height = chunkSize;

                for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
                    const datasetBytesView = new Float32Array(
                        datasetBytesBuffer,
                        i * IMAGE_SIZE * chunkSize * 4,
                        IMAGE_SIZE * chunkSize
                    );

                    ctx.drawImage(
                        img,
                        0, i * chunkSize,
                        img.width, chunkSize,
                        0, 0,
                        img.width, chunkSize
                    );

                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

                    for (let j = 0; j < imageData.data.length / 4; j++) {
                        // Use red channel and normalize to [0, 1]
                        datasetBytesView[j] = imageData.data[j * 4] / 255;
                    }

                    if (onProgress) {
                        this.loadingProgress = Math.floor((i + 1) / (NUM_DATASET_ELEMENTS / chunkSize) * 50);
                        onProgress(this.loadingProgress);
                    }
                }

                this.datasetImages = new Float32Array(datasetBytesBuffer);
                resolve();
            };

            img.onerror = reject;
            img.src = MNIST_IMAGES_SPRITE_PATH;
        });

        // Load labels
        const labelsRequest = fetch(MNIST_LABELS_PATH)
            .then(response => response.arrayBuffer())
            .then(buffer => {
                this.datasetLabels = new Uint8Array(buffer);
                if (onProgress) {
                    this.loadingProgress = 75;
                    onProgress(this.loadingProgress);
                }
            });

        await Promise.all([imgRequest, labelsRequest]);

        // Create shuffled indices
        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

        this.isLoaded = true;
        if (onProgress) {
            this.loadingProgress = 100;
            onProgress(100);
        }

        console.log('MNIST loaded:', NUM_TRAIN_ELEMENTS, 'train,', NUM_TEST_ELEMENTS, 'test images');
    }

    /**
     * Get next training batch
     * @param {number} batchSize
     * @returns {{xs: tf.Tensor, ys: tf.Tensor}}
     */
    nextTrainBatch(batchSize) {
        return this.nextBatch(batchSize, [this.datasetImages, this.datasetLabels], () => {
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
        return this.nextBatch(batchSize, [this.datasetImages, this.datasetLabels], () => {
            this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length;
            return this.testIndices[this.shuffledTestIndex] + NUM_TRAIN_ELEMENTS;
        });
    }

    /**
     * Internal batch fetching
     */
    nextBatch(batchSize, data, indexFn) {
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

        for (let i = 0; i < batchSize; i++) {
            const idx = indexFn();

            const image = data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
            batchImagesArray.set(image, i * IMAGE_SIZE);

            const label = data[1][idx];
            batchLabelsArray[i * NUM_CLASSES + label] = 1;
        }

        const xs = tf.tensor4d(batchImagesArray, [batchSize, 28, 28, 1]);
        const ys = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

        return { xs, ys };
    }

    /**
     * Get a single random sample for visualization
     * @param {boolean} fromTest - Whether to sample from test set
     * @returns {{image: tf.Tensor, label: number}}
     */
    getRandomSample(fromTest = false) {
        const data = fromTest ? this.testIndices : this.trainIndices;
        const offset = fromTest ? NUM_TRAIN_ELEMENTS : 0;
        const idx = data[Math.floor(Math.random() * data.length)] + offset;

        const imageData = this.datasetImages.slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
        const image = tf.tensor4d(imageData, [1, 28, 28, 1]);
        const label = this.datasetLabels[idx];

        return { image, label, imageData };
    }

    /**
     * Get image data as 2D array for canvas rendering
     * @param {Float32Array} imageData - Flat image data
     * @returns {number[][]} 28x28 array
     */
    static imageDataTo2D(imageData) {
        const result = [];
        for (let i = 0; i < 28; i++) {
            const row = [];
            for (let j = 0; j < 28; j++) {
                row.push(imageData[i * 28 + j]);
            }
            result.push(row);
        }
        return result;
    }
}
