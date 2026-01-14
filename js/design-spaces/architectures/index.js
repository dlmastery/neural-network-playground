/**
 * Architecture Index
 * Aggregates and exports all architecture configurations
 */

import { transformers } from './transformers.js';
import { llms } from './llms.js';
import { diffusion } from './diffusion.js';
import { gnns } from './gnns.js';
import { cnns } from './cnns.js';
import { rnns } from './rnns.js';
import { ssms } from './ssms.js';
import { vits } from './vits.js';
import { vaes } from './vaes.js';
import { gans } from './gans.js';

// Export individual architectures for selective imports
export {
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
};

// Export as array for iteration (maintains original order)
export const architectures = [
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
];

// Export as map for lookup by ID
export const architecturesById = {
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
};

// Helper to get architecture by ID
export function getArchitecture(id) {
    return architecturesById[id] || null;
}

// Helper to get all architecture IDs
export function getArchitectureIds() {
    return architectures.map(arch => arch.id);
}
