/**
 * Design Spaces Parameter Data
 *
 * This file re-exports from modular architecture files for backward compatibility.
 * For new code, import directly from the specific modules:
 *
 *   import { parameterCategories } from './categories.js';
 *   import { transformers, llms } from './architectures/index.js';
 *
 * File Structure:
 *   design-spaces/
 *   ├── categories.js           - Parameter category definitions
 *   ├── architectures/
 *   │   ├── index.js           - Aggregates all architectures
 *   │   ├── transformers.js    - Transformer architecture
 *   │   ├── llms.js            - Large Language Models
 *   │   ├── diffusion.js       - Diffusion Models
 *   │   ├── gnns.js            - Graph Neural Networks
 *   │   ├── cnns.js            - Convolutional Neural Networks
 *   │   ├── rnns.js            - RNNs & LSTMs
 *   │   ├── ssms.js            - SSMs / Mamba
 *   │   ├── vits.js            - Vision Transformers
 *   │   ├── vaes.js            - Variational Autoencoders
 *   │   └── gans.js            - GANs
 *   └── design-spaces-data.js  - This file (backward compat re-exports)
 */

// Re-export categories
export { parameterCategories } from './categories.js';

// Re-export architectures
export {
    architectures,
    architecturesById,
    getArchitecture,
    getArchitectureIds,
    transformers,
    llms,
    diffusion,
    gnns,
    cnns,
    rnns,
    ssms,
    vits,
    vaes,
    gans
} from './architectures/index.js';
