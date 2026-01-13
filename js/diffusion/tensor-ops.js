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
