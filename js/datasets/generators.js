/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Dataset Generators - Classic ML Classification Problems
 * ═══════════════════════════════════════════════════════════════════════════
 */

import { Tensor } from '../core/tensor.js';

/**
 * Generate 2D classification datasets
 */
export class DatasetGenerator {
    /**
     * Generate circle dataset
     * Points inside radius are class 1, outside are class 0
     */
    static circle(numPoints = 200, noise = 0) {
        const X = [];
        const y = [];

        for (let i = 0; i < numPoints; i++) {
            // Random point in [-1, 1] x [-1, 1]
            const x1 = Math.random() * 2 - 1;
            const x2 = Math.random() * 2 - 1;

            // Add noise
            const n1 = noise * (Math.random() * 2 - 1);
            const n2 = noise * (Math.random() * 2 - 1);

            // Class based on distance from origin
            const distance = Math.sqrt(x1 * x1 + x2 * x2);
            const label = distance < 0.5 ? 1 : 0;

            X.push([x1 + n1, x2 + n2]);
            y.push([label]);
        }

        return {
            X: Tensor.fromArray(X),
            y: Tensor.fromArray(y),
            name: 'circle'
        };
    }

    /**
     * Generate XOR dataset
     * Classic non-linearly separable problem
     */
    static xor(numPoints = 200, noise = 0) {
        const X = [];
        const y = [];

        for (let i = 0; i < numPoints; i++) {
            // Random point in [-1, 1] x [-1, 1]
            const x1 = Math.random() * 2 - 1;
            const x2 = Math.random() * 2 - 1;

            // Add noise
            const n1 = noise * (Math.random() * 2 - 1);
            const n2 = noise * (Math.random() * 2 - 1);

            // XOR: same sign = 0, different sign = 1
            const label = (x1 * x2 >= 0) ? 0 : 1;

            X.push([x1 + n1, x2 + n2]);
            y.push([label]);
        }

        return {
            X: Tensor.fromArray(X),
            y: Tensor.fromArray(y),
            name: 'xor'
        };
    }

    /**
     * Generate two spirals dataset
     * Challenging non-linear decision boundary
     */
    static spiral(numPoints = 200, noise = 0) {
        const X = [];
        const y = [];
        const pointsPerClass = Math.floor(numPoints / 2);

        for (let classLabel = 0; classLabel < 2; classLabel++) {
            for (let i = 0; i < pointsPerClass; i++) {
                // Angle varies from 0 to 3 * PI
                const angle = (i / pointsPerClass) * Math.PI * 3;
                // Radius increases with angle
                const radius = (i / pointsPerClass) * 0.8 + 0.1;

                // Offset spiral for second class
                const angleOffset = classLabel * Math.PI;

                // Convert to Cartesian coordinates
                let x1 = radius * Math.sin(angle + angleOffset);
                let x2 = radius * Math.cos(angle + angleOffset);

                // Add noise
                x1 += noise * (Math.random() * 2 - 1);
                x2 += noise * (Math.random() * 2 - 1);

                X.push([x1, x2]);
                y.push([classLabel]);
            }
        }

        // Shuffle the data
        const indices = Array.from({ length: X.length }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        const shuffledX = indices.map(i => X[i]);
        const shuffledY = indices.map(i => y[i]);

        return {
            X: Tensor.fromArray(shuffledX),
            y: Tensor.fromArray(shuffledY),
            name: 'spiral'
        };
    }

    /**
     * Generate Gaussian clusters dataset
     */
    static clusters(numPoints = 200, noise = 0) {
        const X = [];
        const y = [];
        const pointsPerClass = Math.floor(numPoints / 2);

        // Cluster centers
        const centers = [
            [-0.5, -0.5],
            [0.5, 0.5]
        ];

        for (let classLabel = 0; classLabel < 2; classLabel++) {
            const [cx, cy] = centers[classLabel];

            for (let i = 0; i < pointsPerClass; i++) {
                // Generate point from Gaussian distribution around center
                const stdDev = 0.25 + noise * 0.2;

                // Box-Muller transform for normal distribution
                const u1 = Math.random();
                const u2 = Math.random();
                const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                const z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);

                const x1 = cx + z1 * stdDev;
                const x2 = cy + z2 * stdDev;

                X.push([x1, x2]);
                y.push([classLabel]);
            }
        }

        // Shuffle
        const indices = Array.from({ length: X.length }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        const shuffledX = indices.map(i => X[i]);
        const shuffledY = indices.map(i => y[i]);

        return {
            X: Tensor.fromArray(shuffledX),
            y: Tensor.fromArray(shuffledY),
            name: 'clusters'
        };
    }

    /**
     * Generate moons dataset
     * Two interleaving half circles
     */
    static moons(numPoints = 200, noise = 0) {
        const X = [];
        const y = [];
        const pointsPerClass = Math.floor(numPoints / 2);

        // First moon (top)
        for (let i = 0; i < pointsPerClass; i++) {
            const angle = Math.PI * i / pointsPerClass;
            const x1 = Math.cos(angle);
            const x2 = Math.sin(angle);

            const n1 = noise * (Math.random() * 2 - 1);
            const n2 = noise * (Math.random() * 2 - 1);

            X.push([x1 + n1, x2 + n2]);
            y.push([0]);
        }

        // Second moon (bottom, offset)
        for (let i = 0; i < pointsPerClass; i++) {
            const angle = Math.PI * i / pointsPerClass;
            const x1 = 1 - Math.cos(angle);
            const x2 = 0.5 - Math.sin(angle);

            const n1 = noise * (Math.random() * 2 - 1);
            const n2 = noise * (Math.random() * 2 - 1);

            X.push([x1 + n1, x2 + n2]);
            y.push([1]);
        }

        // Normalize to [-1, 1]
        const normalizedX = X.map(([x1, x2]) => [
            (x1 - 0.5) * 1.5,
            (x2 - 0.25) * 2
        ]);

        // Shuffle
        const indices = Array.from({ length: normalizedX.length }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        const shuffledX = indices.map(i => normalizedX[i]);
        const shuffledY = indices.map(i => y[i]);

        return {
            X: Tensor.fromArray(shuffledX),
            y: Tensor.fromArray(shuffledY),
            name: 'moons'
        };
    }

    /**
     * Generate dataset by name
     */
    static generate(name, numPoints = 200, noise = 0) {
        switch (name) {
            case 'circle':
                return DatasetGenerator.circle(numPoints, noise);
            case 'xor':
                return DatasetGenerator.xor(numPoints, noise);
            case 'spiral':
                return DatasetGenerator.spiral(numPoints, noise);
            case 'clusters':
                return DatasetGenerator.clusters(numPoints, noise);
            case 'moons':
                return DatasetGenerator.moons(numPoints, noise);
            default:
                return DatasetGenerator.circle(numPoints, noise);
        }
    }

    /**
     * Get a batch of random samples
     */
    static getBatch(X, y, batchSize) {
        if (batchSize >= X.rows || batchSize === 'all') {
            return { X, y };
        }

        // Random indices
        const indices = [];
        for (let i = 0; i < batchSize; i++) {
            indices.push(Math.floor(Math.random() * X.rows));
        }

        return {
            X: X.getRows(indices),
            y: y.getRows(indices)
        };
    }

    /**
     * Split dataset into train/test sets
     */
    static trainTestSplit(X, y, testRatio = 0.2) {
        const numTest = Math.floor(X.rows * testRatio);
        const numTrain = X.rows - numTest;

        // Shuffle indices
        const indices = Array.from({ length: X.rows }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        const trainIndices = indices.slice(0, numTrain);
        const testIndices = indices.slice(numTrain);

        return {
            trainX: X.getRows(trainIndices),
            trainY: y.getRows(trainIndices),
            testX: X.getRows(testIndices),
            testY: y.getRows(testIndices)
        };
    }
}
