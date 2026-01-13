/**
 * GNN Dataset
 * Graph datasets for Graph Neural Network visualization
 * Includes Zachary's Karate Club and synthetic graph generator
 */

/**
 * Zachary's Karate Club Network
 * Classic social network dataset with 34 nodes and 78 edges
 * Two factions: Mr. Hi (instructor) and Officer (club president)
 */
export const KARATE_CLUB = {
    name: 'Karate Club',
    numNodes: 34,
    numClasses: 2,
    classNames: ['Mr. Hi', 'Officer'],
    // All 78 edges (undirected, stored once)
    edges: [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8],
        [0, 10], [0, 11], [0, 12], [0, 13], [0, 17], [0, 19], [0, 21], [0, 31],
        [1, 2], [1, 3], [1, 7], [1, 13], [1, 17], [1, 19], [1, 21], [1, 30],
        [2, 3], [2, 7], [2, 8], [2, 9], [2, 13], [2, 27], [2, 28], [2, 32],
        [3, 7], [3, 12], [3, 13],
        [4, 6], [4, 10],
        [5, 6], [5, 10], [5, 16],
        [6, 16],
        [8, 30], [8, 32], [8, 33],
        [9, 33],
        [13, 33],
        [14, 32], [14, 33],
        [15, 32], [15, 33],
        [18, 32], [18, 33],
        [19, 33],
        [20, 32], [20, 33],
        [22, 32], [22, 33],
        [23, 25], [23, 27], [23, 29], [23, 32], [23, 33],
        [24, 25], [24, 27], [24, 31],
        [25, 31],
        [26, 29], [26, 33],
        [27, 33],
        [28, 31], [28, 33],
        [29, 32], [29, 33],
        [30, 32], [30, 33],
        [31, 32], [31, 33],
        [32, 33]
    ],
    // Ground truth labels: 0 = Mr. Hi faction, 1 = Officer faction
    labels: [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1,  // nodes 0-9
        0, 0, 0, 0, 1, 1, 0, 0, 1, 0,  // nodes 10-19
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1,  // nodes 20-29
        1, 1, 1, 1                      // nodes 30-33
    ]
};

/**
 * Generate a synthetic graph with community structure
 */
export function generateSyntheticGraph(numNodes = 20, numClasses = 2, edgeProbIntra = 0.6, edgeProbInter = 0.1) {
    const nodesPerClass = Math.floor(numNodes / numClasses);
    const labels = [];
    const edges = [];

    // Assign labels
    for (let i = 0; i < numNodes; i++) {
        labels.push(Math.floor(i / nodesPerClass) % numClasses);
    }

    // Generate edges
    for (let i = 0; i < numNodes; i++) {
        for (let j = i + 1; j < numNodes; j++) {
            const sameClass = labels[i] === labels[j];
            const prob = sameClass ? edgeProbIntra : edgeProbInter;
            if (Math.random() < prob) {
                edges.push([i, j]);
            }
        }
    }

    return {
        name: 'Synthetic',
        numNodes,
        numClasses,
        classNames: Array.from({ length: numClasses }, (_, i) => `Class ${i}`),
        edges,
        labels
    };
}

/**
 * GNN Dataset class - wraps graph data with utility methods
 */
export class GNNDataset {
    constructor(graphData) {
        this.name = graphData.name;
        this.numNodes = graphData.numNodes;
        this.numClasses = graphData.numClasses;
        this.classNames = graphData.classNames;
        this.edges = graphData.edges;
        this.labels = graphData.labels;

        // Build adjacency structures
        this.neighbors = this.buildNeighborList();
        this.adjMatrix = this.buildAdjMatrix();
        this.adjNormalized = this.normalizeAdjMatrix();

        // Initialize node features (one-hot encoding of node ID)
        this.nodeFeatures = this.initFeatures();
    }

    /**
     * Build neighbor list from edges
     */
    buildNeighborList() {
        const neighbors = Array.from({ length: this.numNodes }, () => []);

        for (const [i, j] of this.edges) {
            neighbors[i].push(j);
            neighbors[j].push(i);
        }

        return neighbors;
    }

    /**
     * Build adjacency matrix from edges
     * Includes self-loops (A + I)
     */
    buildAdjMatrix() {
        const n = this.numNodes;
        const adj = new Float32Array(n * n);

        // Add self-loops
        for (let i = 0; i < n; i++) {
            adj[i * n + i] = 1;
        }

        // Add edges (symmetric)
        for (const [i, j] of this.edges) {
            adj[i * n + j] = 1;
            adj[j * n + i] = 1;
        }

        return adj;
    }

    /**
     * Normalize adjacency matrix for GCN
     * D^(-1/2) * A * D^(-1/2) where A includes self-loops
     */
    normalizeAdjMatrix() {
        const n = this.numNodes;
        const adj = this.adjMatrix;
        const normalized = new Float32Array(n * n);

        // Compute degree (sum of row in adjacency with self-loops)
        const degree = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let j = 0; j < n; j++) {
                sum += adj[i * n + j];
            }
            degree[i] = sum;
        }

        // Compute D^(-1/2)
        const degInvSqrt = degree.map(d => d > 0 ? 1 / Math.sqrt(d) : 0);

        // Compute D^(-1/2) * A * D^(-1/2)
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                normalized[i * n + j] = degInvSqrt[i] * adj[i * n + j] * degInvSqrt[j];
            }
        }

        return normalized;
    }

    /**
     * Initialize node features
     * Default: one-hot encoding of node ID
     */
    initFeatures() {
        const n = this.numNodes;
        const features = new Float32Array(n * n);

        // One-hot encoding: feature[i][i] = 1
        for (let i = 0; i < n; i++) {
            features[i * n + i] = 1;
        }

        return features;
    }

    /**
     * Get feature dimension
     */
    get featureDim() {
        return this.numNodes; // One-hot encoding
    }

    /**
     * Get number of edges
     */
    get numEdges() {
        return this.edges.length;
    }

    /**
     * Get neighbors of a node
     */
    getNeighbors(nodeId) {
        return this.neighbors[nodeId] || [];
    }

    /**
     * Get degree of a node
     */
    getDegree(nodeId) {
        return this.neighbors[nodeId]?.length || 0;
    }

    /**
     * Get node features as 2D array-like accessor
     */
    getNodeFeature(nodeId) {
        const n = this.numNodes;
        const start = nodeId * n;
        return Array.from(this.nodeFeatures.slice(start, start + n));
    }

    /**
     * Get adjacency row for a node
     */
    getAdjRow(nodeId) {
        const n = this.numNodes;
        const start = nodeId * n;
        return this.adjNormalized.slice(start, start + n);
    }

    /**
     * Convert flat features to 2D array for visualization
     */
    getFeaturesAs2D() {
        const n = this.numNodes;
        const result = [];
        for (let i = 0; i < n; i++) {
            result.push(this.getNodeFeature(i));
        }
        return result;
    }

    /**
     * Get train/test split masks (random split)
     */
    getTrainTestSplit(trainRatio = 0.7) {
        const n = this.numNodes;
        const indices = Array.from({ length: n }, (_, i) => i);

        // Shuffle
        for (let i = n - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        const trainSize = Math.floor(n * trainRatio);
        const trainMask = new Array(n).fill(false);
        const testMask = new Array(n).fill(false);

        for (let i = 0; i < trainSize; i++) {
            trainMask[indices[i]] = true;
        }
        for (let i = trainSize; i < n; i++) {
            testMask[indices[i]] = true;
        }

        return { trainMask, testMask };
    }
}

/**
 * Available datasets
 */
export const DATASETS = {
    karate: KARATE_CLUB,
    synthetic: () => generateSyntheticGraph(20, 2, 0.5, 0.1)
};

/**
 * Load a dataset by name
 */
export function loadDataset(name) {
    const data = DATASETS[name];
    if (!data) {
        throw new Error(`Unknown dataset: ${name}`);
    }

    const graphData = typeof data === 'function' ? data() : data;
    return new GNNDataset(graphData);
}
