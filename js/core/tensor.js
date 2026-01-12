/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Tensor Class - High-Performance Matrix Operations for Neural Networks
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * A lightweight tensor implementation using Float32Array for optimal performance.
 * Supports all operations needed for forward/backward passes in neural networks.
 */

export class Tensor {
    /**
     * Create a new Tensor
     * @param {number} rows - Number of rows
     * @param {number} cols - Number of columns
     * @param {Float32Array|Array|null} data - Initial data (optional)
     */
    constructor(rows, cols, data = null) {
        this.rows = rows;
        this.cols = cols;
        this.size = rows * cols;

        if (data instanceof Float32Array) {
            this.data = data;
        } else if (Array.isArray(data)) {
            this.data = new Float32Array(data.flat());
        } else {
            this.data = new Float32Array(this.size);
        }
    }

    /**
     * Get value at position (i, j)
     */
    get(i, j) {
        return this.data[i * this.cols + j];
    }

    /**
     * Set value at position (i, j)
     */
    set(i, j, value) {
        this.data[i * this.cols + j] = value;
    }

    /**
     * Create a tensor filled with zeros
     */
    static zeros(rows, cols) {
        return new Tensor(rows, cols);
    }

    /**
     * Create a tensor filled with ones
     */
    static ones(rows, cols) {
        const t = new Tensor(rows, cols);
        t.data.fill(1);
        return t;
    }

    /**
     * Create tensor with random values (Xavier/Glorot initialization)
     * Optimal for tanh activations
     */
    static random(rows, cols, scale = null) {
        const t = new Tensor(rows, cols);
        // Xavier initialization: scale by sqrt(2 / (fan_in + fan_out))
        const s = scale ?? Math.sqrt(2 / (rows + cols));
        for (let i = 0; i < t.size; i++) {
            // Box-Muller transform for normal distribution
            const u1 = Math.random();
            const u2 = Math.random();
            const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            t.data[i] = z * s;
        }
        return t;
    }

    /**
     * Create tensor with He initialization
     * Optimal for ReLU activations
     */
    static randomHe(rows, cols) {
        const scale = Math.sqrt(2 / rows);
        return Tensor.random(rows, cols, scale);
    }

    /**
     * Create tensor from 2D array
     */
    static fromArray(arr) {
        const rows = arr.length;
        const cols = arr[0].length;
        const t = new Tensor(rows, cols);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                t.set(i, j, arr[i][j]);
            }
        }
        return t;
    }

    /**
     * Create a row vector from 1D array
     */
    static fromVector(arr) {
        return new Tensor(1, arr.length, new Float32Array(arr));
    }

    /**
     * Clone this tensor
     */
    clone() {
        return new Tensor(this.rows, this.cols, new Float32Array(this.data));
    }

    /**
     * Convert to 2D array
     */
    toArray() {
        const result = [];
        for (let i = 0; i < this.rows; i++) {
            result.push([]);
            for (let j = 0; j < this.cols; j++) {
                result[i].push(this.get(i, j));
            }
        }
        return result;
    }

    /**
     * Matrix multiplication: this @ other
     */
    matmul(other) {
        if (this.cols !== other.rows) {
            throw new Error(`Matrix dimensions don't match: (${this.rows}x${this.cols}) @ (${other.rows}x${other.cols})`);
        }

        const result = new Tensor(this.rows, other.cols);

        // Optimized matrix multiplication with cache-friendly access
        for (let i = 0; i < this.rows; i++) {
            for (let k = 0; k < this.cols; k++) {
                const aik = this.data[i * this.cols + k];
                for (let j = 0; j < other.cols; j++) {
                    result.data[i * other.cols + j] += aik * other.data[k * other.cols + j];
                }
            }
        }

        return result;
    }

    /**
     * Transpose matrix
     */
    transpose() {
        const result = new Tensor(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[j * this.rows + i] = this.data[i * this.cols + j];
            }
        }
        return result;
    }

    /**
     * Element-wise addition
     */
    add(other) {
        if (other instanceof Tensor) {
            // Broadcasting for bias addition (other is 1 x cols)
            if (other.rows === 1 && other.cols === this.cols) {
                const result = this.clone();
                for (let i = 0; i < this.rows; i++) {
                    for (let j = 0; j < this.cols; j++) {
                        result.data[i * this.cols + j] += other.data[j];
                    }
                }
                return result;
            }

            // Standard element-wise addition
            if (this.rows !== other.rows || this.cols !== other.cols) {
                throw new Error(`Tensor dimensions don't match for addition`);
            }
            const result = new Tensor(this.rows, this.cols);
            for (let i = 0; i < this.size; i++) {
                result.data[i] = this.data[i] + other.data[i];
            }
            return result;
        }

        // Scalar addition
        const result = this.clone();
        for (let i = 0; i < this.size; i++) {
            result.data[i] += other;
        }
        return result;
    }

    /**
     * Element-wise subtraction
     */
    sub(other) {
        if (other instanceof Tensor) {
            if (this.rows !== other.rows || this.cols !== other.cols) {
                throw new Error(`Tensor dimensions don't match for subtraction`);
            }
            const result = new Tensor(this.rows, this.cols);
            for (let i = 0; i < this.size; i++) {
                result.data[i] = this.data[i] - other.data[i];
            }
            return result;
        }

        const result = this.clone();
        for (let i = 0; i < this.size; i++) {
            result.data[i] -= other;
        }
        return result;
    }

    /**
     * Element-wise multiplication (Hadamard product)
     */
    mul(other) {
        if (other instanceof Tensor) {
            if (this.rows !== other.rows || this.cols !== other.cols) {
                throw new Error(`Tensor dimensions don't match for multiplication`);
            }
            const result = new Tensor(this.rows, this.cols);
            for (let i = 0; i < this.size; i++) {
                result.data[i] = this.data[i] * other.data[i];
            }
            return result;
        }

        // Scalar multiplication
        const result = new Tensor(this.rows, this.cols);
        for (let i = 0; i < this.size; i++) {
            result.data[i] = this.data[i] * other;
        }
        return result;
    }

    /**
     * Element-wise division
     */
    div(other) {
        if (other instanceof Tensor) {
            const result = new Tensor(this.rows, this.cols);
            for (let i = 0; i < this.size; i++) {
                result.data[i] = this.data[i] / other.data[i];
            }
            return result;
        }

        const result = new Tensor(this.rows, this.cols);
        for (let i = 0; i < this.size; i++) {
            result.data[i] = this.data[i] / other;
        }
        return result;
    }

    /**
     * Apply function element-wise
     */
    map(fn) {
        const result = new Tensor(this.rows, this.cols);
        for (let i = 0; i < this.size; i++) {
            result.data[i] = fn(this.data[i], i);
        }
        return result;
    }

    /**
     * Modify in place
     */
    mapInPlace(fn) {
        for (let i = 0; i < this.size; i++) {
            this.data[i] = fn(this.data[i], i);
        }
        return this;
    }

    /**
     * Sum all elements
     */
    sum() {
        let total = 0;
        for (let i = 0; i < this.size; i++) {
            total += this.data[i];
        }
        return total;
    }

    /**
     * Sum along axis (0 = columns, 1 = rows)
     */
    sumAxis(axis) {
        if (axis === 0) {
            // Sum columns → 1 x cols
            const result = new Tensor(1, this.cols);
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    result.data[j] += this.data[i * this.cols + j];
                }
            }
            return result;
        } else {
            // Sum rows → rows x 1
            const result = new Tensor(this.rows, 1);
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    result.data[i] += this.data[i * this.cols + j];
                }
            }
            return result;
        }
    }

    /**
     * Mean of all elements
     */
    mean() {
        return this.sum() / this.size;
    }

    /**
     * Max value
     */
    max() {
        let maxVal = -Infinity;
        for (let i = 0; i < this.size; i++) {
            if (this.data[i] > maxVal) maxVal = this.data[i];
        }
        return maxVal;
    }

    /**
     * Min value
     */
    min() {
        let minVal = Infinity;
        for (let i = 0; i < this.size; i++) {
            if (this.data[i] < minVal) minVal = this.data[i];
        }
        return minVal;
    }

    /**
     * Element-wise power
     */
    pow(exp) {
        return this.map(x => Math.pow(x, exp));
    }

    /**
     * Element-wise square root
     */
    sqrt() {
        return this.map(x => Math.sqrt(x));
    }

    /**
     * Element-wise exponential
     */
    exp() {
        return this.map(x => Math.exp(x));
    }

    /**
     * Element-wise natural log
     */
    log() {
        return this.map(x => Math.log(x + 1e-8)); // Add small epsilon to avoid log(0)
    }

    /**
     * Clip values to range
     */
    clip(min, max) {
        return this.map(x => Math.max(min, Math.min(max, x)));
    }

    /**
     * Get a specific row as a new tensor
     */
    row(i) {
        const result = new Tensor(1, this.cols);
        for (let j = 0; j < this.cols; j++) {
            result.data[j] = this.data[i * this.cols + j];
        }
        return result;
    }

    /**
     * Get specific rows (for batch selection)
     */
    getRows(indices) {
        const result = new Tensor(indices.length, this.cols);
        for (let i = 0; i < indices.length; i++) {
            const srcIdx = indices[i] * this.cols;
            const dstIdx = i * this.cols;
            for (let j = 0; j < this.cols; j++) {
                result.data[dstIdx + j] = this.data[srcIdx + j];
            }
        }
        return result;
    }

    /**
     * Concatenate tensors vertically (stack rows)
     */
    static vstack(tensors) {
        const cols = tensors[0].cols;
        let totalRows = 0;
        for (const t of tensors) {
            if (t.cols !== cols) throw new Error('All tensors must have same number of columns');
            totalRows += t.rows;
        }

        const result = new Tensor(totalRows, cols);
        let offset = 0;
        for (const t of tensors) {
            result.data.set(t.data, offset);
            offset += t.size;
        }
        return result;
    }

    /**
     * String representation for debugging
     */
    toString() {
        let str = `Tensor(${this.rows}x${this.cols}):\n`;
        for (let i = 0; i < Math.min(this.rows, 10); i++) {
            str += '  [';
            for (let j = 0; j < Math.min(this.cols, 10); j++) {
                str += this.get(i, j).toFixed(4);
                if (j < this.cols - 1) str += ', ';
            }
            if (this.cols > 10) str += ', ...';
            str += ']\n';
        }
        if (this.rows > 10) str += '  ...\n';
        return str;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Activation Functions
// ═══════════════════════════════════════════════════════════════════════════

export const Activations = {
    // ReLU
    relu: {
        forward: (x) => x.map(v => Math.max(0, v)),
        backward: (x) => x.map(v => v > 0 ? 1 : 0)
    },

    // Leaky ReLU
    leakyRelu: {
        forward: (x, alpha = 0.01) => x.map(v => v > 0 ? v : alpha * v),
        backward: (x, alpha = 0.01) => x.map(v => v > 0 ? 1 : alpha)
    },

    // Sigmoid
    sigmoid: {
        forward: (x) => x.map(v => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, v))))),
        backward: (x) => {
            const s = Activations.sigmoid.forward(x);
            return s.mul(s.map(v => 1 - v));
        }
    },

    // Tanh
    tanh: {
        forward: (x) => x.map(v => Math.tanh(v)),
        backward: (x) => x.map(v => 1 - Math.pow(Math.tanh(v), 2))
    },

    // Swish (SiLU)
    swish: {
        forward: (x) => {
            return x.map(v => v / (1 + Math.exp(-v)));
        },
        backward: (x) => {
            return x.map(v => {
                const sig = 1 / (1 + Math.exp(-v));
                return sig + v * sig * (1 - sig);
            });
        }
    },

    // Linear (identity)
    linear: {
        forward: (x) => x.clone(),
        backward: (x) => Tensor.ones(x.rows, x.cols)
    },

    // Softmax (for output layer)
    softmax: {
        forward: (x) => {
            const result = new Tensor(x.rows, x.cols);
            for (let i = 0; i < x.rows; i++) {
                // Subtract max for numerical stability
                let maxVal = -Infinity;
                for (let j = 0; j < x.cols; j++) {
                    const val = x.get(i, j);
                    if (val > maxVal) maxVal = val;
                }

                let sum = 0;
                for (let j = 0; j < x.cols; j++) {
                    const expVal = Math.exp(x.get(i, j) - maxVal);
                    result.set(i, j, expVal);
                    sum += expVal;
                }

                for (let j = 0; j < x.cols; j++) {
                    result.set(i, j, result.get(i, j) / sum);
                }
            }
            return result;
        },
        // Softmax derivative is handled specially in cross-entropy loss
        backward: (x) => Tensor.ones(x.rows, x.cols)
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Loss Functions
// ═══════════════════════════════════════════════════════════════════════════

export const Losses = {
    // Mean Squared Error
    mse: {
        forward: (predicted, target) => {
            const diff = predicted.sub(target);
            return diff.mul(diff).mean();
        },
        backward: (predicted, target) => {
            return predicted.sub(target).mul(2 / predicted.size);
        }
    },

    // Binary Cross-Entropy
    binaryCrossEntropy: {
        forward: (predicted, target) => {
            const eps = 1e-8;
            const clipped = predicted.clip(eps, 1 - eps);
            let loss = 0;
            for (let i = 0; i < target.size; i++) {
                const y = target.data[i];
                const p = clipped.data[i];
                loss -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
            }
            return loss / target.rows;
        },
        backward: (predicted, target) => {
            const eps = 1e-8;
            const clipped = predicted.clip(eps, 1 - eps);
            const result = new Tensor(predicted.rows, predicted.cols);
            for (let i = 0; i < result.size; i++) {
                const y = target.data[i];
                const p = clipped.data[i];
                result.data[i] = (-y / p + (1 - y) / (1 - p)) / target.rows;
            }
            return result;
        }
    },

    // Cross-Entropy (for softmax output)
    crossEntropy: {
        forward: (predicted, target) => {
            const eps = 1e-8;
            let loss = 0;
            for (let i = 0; i < target.size; i++) {
                if (target.data[i] > 0) {
                    loss -= target.data[i] * Math.log(predicted.data[i] + eps);
                }
            }
            return loss / target.rows;
        },
        backward: (predicted, target) => {
            // For softmax + cross-entropy, gradient simplifies to (predicted - target)
            return predicted.sub(target).mul(1 / target.rows);
        }
    }
};
