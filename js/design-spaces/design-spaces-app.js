/**
 * Design Spaces App
 * Main controller for the Design Spaces tab
 */

import { architectures } from './design-spaces-data.js';

export class DesignSpacesApp {
    constructor() {
        this.isInitialized = false;
        this.currentArch = 'transformers';
        this.elements = {};
        this.parameterValues = {};
    }

    /**
     * Initialize the design spaces tab
     */
    async init() {
        if (this.isInitialized) return;

        console.log('[DesignSpaces] Initializing...');

        this.cacheElements();
        this.setupEventListeners();
        this.renderArchitectureNav();
        this.selectArchitecture('transformers');

        this.isInitialized = true;
        console.log('[DesignSpaces] Initialized');
    }

    /**
     * Cache DOM element references
     */
    cacheElements() {
        this.elements = {
            currentArch: document.getElementById('design-spaces-current-arch'),
            searchInput: document.getElementById('design-spaces-search'),
            viewCards: document.getElementById('design-spaces-view-cards'),
            viewTable: document.getElementById('design-spaces-view-table'),
            copyBtn: document.getElementById('design-spaces-copy-btn'),
            archList: document.getElementById('design-spaces-arch-list'),
            archTitle: document.getElementById('design-spaces-arch-title'),
            archDesc: document.getElementById('design-spaces-arch-desc'),
            paramsGrid: document.getElementById('design-spaces-params-grid'),
            whenToUse: document.getElementById('design-spaces-when-to-use'),
            innovation: document.getElementById('design-spaces-innovation'),
            size: document.getElementById('design-spaces-size')
        };
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Search
        this.elements.searchInput?.addEventListener('input', (e) => {
            this.filterParameters(e.target.value);
        });

        // View toggle
        this.elements.viewCards?.addEventListener('click', () => this.setView('cards'));
        this.elements.viewTable?.addEventListener('click', () => this.setView('table'));

        // Copy config
        this.elements.copyBtn?.addEventListener('click', () => this.copyConfig());
    }

    /**
     * Render the architecture navigation sidebar
     */
    renderArchitectureNav() {
        if (!this.elements.archList) return;

        const html = architectures.map(arch => `
            <div class="design-spaces-arch-item ${arch.id === this.currentArch ? 'design-spaces-arch-item--active' : ''}"
                 data-arch="${arch.id}">
                <div class="design-spaces-arch-icon" style="background: ${arch.color}20; color: ${arch.color}">
                    ${arch.icon}
                </div>
                <div>
                    <div class="design-spaces-arch-name">${arch.name}</div>
                    <div class="design-spaces-arch-subtitle">${arch.yearIntroduced}</div>
                </div>
            </div>
        `).join('');

        this.elements.archList.innerHTML = html;

        // Add click listeners
        this.elements.archList.querySelectorAll('.design-spaces-arch-item').forEach(item => {
            item.addEventListener('click', () => {
                this.selectArchitecture(item.dataset.arch);
            });
        });
    }

    /**
     * Select an architecture and display its parameters
     */
    selectArchitecture(archId) {
        this.currentArch = archId;
        const arch = architectures.find(a => a.id === archId);
        if (!arch) return;

        // Update panel data attribute for CSS color theming
        const panel = document.getElementById('design-spaces-panel');
        if (panel) {
            panel.setAttribute('data-arch', archId);
        }

        // Update navigation
        this.elements.archList?.querySelectorAll('.design-spaces-arch-item').forEach(item => {
            item.classList.toggle('design-spaces-arch-item--active', item.dataset.arch === archId);
        });

        // Update header
        if (this.elements.currentArch) this.elements.currentArch.textContent = arch.name;
        if (this.elements.archTitle) this.elements.archTitle.textContent = arch.name;
        if (this.elements.archDesc) this.elements.archDesc.textContent = arch.description;

        // Update quick reference
        if (this.elements.whenToUse) this.elements.whenToUse.textContent = arch.whenToUse;
        if (this.elements.innovation) this.elements.innovation.textContent = arch.innovation;
        if (this.elements.size) this.elements.size.textContent = arch.typicalSize;

        // Initialize parameter values
        this.parameterValues[archId] = this.parameterValues[archId] || {};
        arch.parameters.forEach(param => {
            if (this.parameterValues[archId][param.key] === undefined) {
                this.parameterValues[archId][param.key] = param.default;
            }
        });

        // Render parameters
        this.renderParameters(arch);
    }

    /**
     * Render parameter cards for an architecture
     */
    renderParameters(arch) {
        if (!this.elements.paramsGrid) return;

        const html = arch.parameters.map(param =>
            this.renderParameterCard(param, arch)
        ).join('');

        this.elements.paramsGrid.innerHTML = html;

        // Add event listeners to controls
        this.elements.paramsGrid.querySelectorAll('.param-card').forEach(card => {
            const key = card.dataset.key;
            const input = card.querySelector('input, select');

            if (input) {
                input.addEventListener('input', (e) => {
                    this.updateParameterValue(arch.id, key, e.target.value, card);
                });
                input.addEventListener('change', (e) => {
                    this.updateParameterValue(arch.id, key, e.target.value, card);
                });
            }
        });
    }

    /**
     * Render a single parameter card
     */
    renderParameterCard(param, arch) {
        const value = this.parameterValues[arch.id][param.key];
        const controlHtml = this.renderControl(param, value);

        return `
            <div class="param-card" data-key="${param.key}">
                <div class="param-card__header">
                    <span class="param-card__name">
                        ${param.name}
                        <span class="param-card__help" title="${param.tip}">?</span>
                    </span>
                </div>
                <div class="param-card__control">
                    ${controlHtml}
                </div>
                <div class="param-card__tip">
                    <div class="param-card__tip-label">Best Practice</div>
                    <div class="param-card__tip-text">${param.tip}</div>
                </div>
                <div class="param-card__indicators">
                    <span class="param-card__indicator">Compute: ${param.compute}</span>
                    <span class="param-card__indicator">Impact: ${param.impact}</span>
                </div>
            </div>
        `;
    }

    /**
     * Render the appropriate control for a parameter
     */
    renderControl(param, value) {
        switch (param.type) {
            case 'slider':
                return `
                    <div class="param-card__slider-row">
                        <input type="range"
                               class="param-card__slider slider"
                               min="${param.min}"
                               max="${param.max}"
                               step="${param.step}"
                               value="${value}">
                        <span class="param-card__value">${value}</span>
                    </div>
                `;

            case 'dropdown':
                const options = param.options.map(opt =>
                    `<option value="${opt}" ${opt === value ? 'selected' : ''}>${opt}</option>`
                ).join('');
                return `<select class="param-card__select select">${options}</select>`;

            case 'toggle':
                return `
                    <label class="toggle-label">
                        <input type="checkbox" class="toggle-input" ${value ? 'checked' : ''}>
                        <span class="toggle-text">${value ? 'Enabled' : 'Disabled'}</span>
                    </label>
                `;

            case 'computed':
                return `
                    <div class="param-card__computed">
                        <span class="param-card__value">${value}</span>
                        <span class="param-card__formula">(${param.formula})</span>
                    </div>
                `;

            default:
                return `<span>${value}</span>`;
        }
    }

    /**
     * Update a parameter value
     */
    updateParameterValue(archId, key, value, card) {
        const input = card.querySelector('input, select');

        // Handle checkbox specially
        if (input?.type === 'checkbox') {
            value = input.checked;
            const text = card.querySelector('.toggle-text');
            if (text) text.textContent = value ? 'Enabled' : 'Disabled';
        }

        this.parameterValues[archId][key] = value;

        // Update displayed value for sliders
        const valueDisplay = card.querySelector('.param-card__value');
        if (valueDisplay && input?.type === 'range') {
            valueDisplay.textContent = value;
        }
    }

    /**
     * Filter parameters by search query
     */
    filterParameters(query) {
        const q = query.toLowerCase();
        this.elements.paramsGrid?.querySelectorAll('.param-card').forEach(card => {
            const name = card.querySelector('.param-card__name')?.textContent.toLowerCase() || '';
            const tip = card.querySelector('.param-card__tip-text')?.textContent.toLowerCase() || '';
            const matches = name.includes(q) || tip.includes(q);
            card.style.display = matches ? '' : 'none';
        });
    }

    /**
     * Set the view mode (cards or table)
     */
    setView(mode) {
        this.elements.viewCards?.classList.toggle('btn--active', mode === 'cards');
        this.elements.viewTable?.classList.toggle('btn--active', mode === 'table');
        this.elements.paramsGrid?.classList.toggle('design-spaces-params-grid--table', mode === 'table');
    }

    /**
     * Copy current configuration to clipboard
     */
    async copyConfig() {
        const arch = architectures.find(a => a.id === this.currentArch);
        if (!arch) return;

        const config = {
            architecture: arch.name,
            parameters: {}
        };

        arch.parameters.forEach(param => {
            config.parameters[param.key] = this.parameterValues[this.currentArch][param.key];
        });

        const text = JSON.stringify(config, null, 2);

        try {
            await navigator.clipboard.writeText(text);
            const btn = this.elements.copyBtn;
            const originalHTML = btn.innerHTML;
            btn.textContent = 'Copied!';
            setTimeout(() => {
                btn.innerHTML = originalHTML;
            }, 2000);
        } catch (err) {
            console.error('[DesignSpaces] Failed to copy:', err);
        }
    }

    /**
     * Cleanup when leaving tab
     */
    cleanup() {
        // Nothing to clean up for now
    }
}

// Export singleton
export const designSpacesApp = new DesignSpacesApp();
