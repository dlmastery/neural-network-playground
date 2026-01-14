/**
 * Design Spaces App
 * Main controller for the Design Spaces tab
 * Enhanced to display research-backed parameter information
 *
 * Security Note: All HTML content is generated from controlled internal data
 * (design-spaces-data.js), not from user input, so innerHTML usage is safe here.
 */

import { architectures, parameterCategories } from './design-spaces-data.js';

export class DesignSpacesApp {
    constructor() {
        this.isInitialized = false;
        this.currentArch = 'transformers';
        this.elements = {};
        this.parameterValues = {};
        this.expandedCards = new Set();
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
            size: document.getElementById('design-spaces-size'),
            keyPapers: document.getElementById('design-spaces-key-papers'),
            scalingGuide: document.getElementById('design-spaces-scaling-guide')
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
     * Note: Content is from controlled internal data, safe for innerHTML
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
        this.expandedCards.clear();
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
        if (this.elements.archDesc) this.elements.archDesc.textContent = arch.fullDescription || arch.description;

        // Update quick reference
        if (this.elements.whenToUse) this.elements.whenToUse.textContent = arch.whenToUse;
        if (this.elements.innovation) this.elements.innovation.textContent = arch.innovation;
        if (this.elements.size) this.elements.size.textContent = arch.typicalSize;

        // Render key papers if element exists
        this.renderKeyPapers(arch);

        // Render scaling guidelines if element exists
        this.renderScalingGuide(arch);

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
     * Render key papers section
     * Note: Content is from controlled internal data
     */
    renderKeyPapers(arch) {
        if (!this.elements.keyPapers || !arch.keyPapers) return;

        const html = arch.keyPapers.map(paper => `
            <div class="key-paper">
                <div class="key-paper__title">${paper.title}</div>
                <div class="key-paper__meta">${paper.authors}, ${paper.year}</div>
                ${paper.arxiv ? `<a class="key-paper__link" href="https://arxiv.org/abs/${paper.arxiv}" target="_blank" rel="noopener">arXiv:${paper.arxiv}</a>` : ''}
            </div>
        `).join('');

        this.elements.keyPapers.innerHTML = html;
    }

    /**
     * Render scaling guidelines section
     * Note: Content is from controlled internal data
     */
    renderScalingGuide(arch) {
        if (!this.elements.scalingGuide || !arch.scalingGuidelines) return;

        const html = Object.entries(arch.scalingGuidelines).map(([size, guide]) => `
            <div class="scaling-tier">
                <div class="scaling-tier__label">${size.charAt(0).toUpperCase() + size.slice(1)}</div>
                <div class="scaling-tier__details">
                    ${guide.params ? `<span class="scaling-tier__params">${guide.params}</span>` : ''}
                    ${guide.layers ? `<span class="scaling-tier__layers">${guide.layers} layers</span>` : ''}
                    ${guide.tokens ? `<span class="scaling-tier__tokens">${guide.tokens} tokens</span>` : ''}
                </div>
                <div class="scaling-tier__note">${guide.note}</div>
            </div>
        `).join('');

        this.elements.scalingGuide.innerHTML = html;
    }

    /**
     * Render parameter cards grouped by category
     * Note: Content is from controlled internal data
     */
    renderParameters(arch) {
        if (!this.elements.paramsGrid) return;

        // Group parameters by category
        const grouped = {};
        arch.parameters.forEach(param => {
            const cat = param.category || 'other';
            if (!grouped[cat]) grouped[cat] = [];
            grouped[cat].push(param);
        });

        // Render grouped parameters
        let html = '';

        // Define category order
        const categoryOrder = ['architecture', 'attention', 'training', 'regularization', 'efficiency', 'generation', 'other'];

        categoryOrder.forEach(catKey => {
            const params = grouped[catKey];
            if (!params || params.length === 0) return;

            const category = parameterCategories[catKey] || { name: 'Other', icon: '📦' };

            html += `
                <div class="param-category">
                    <div class="param-category__header">
                        <span class="param-category__icon">${category.icon}</span>
                        <span class="param-category__name">${category.name}</span>
                        <span class="param-category__count">${params.length}</span>
                    </div>
                    <div class="param-category__grid">
                        ${params.map(param => this.renderParameterCard(param, arch)).join('')}
                    </div>
                </div>
            `;
        });

        this.elements.paramsGrid.innerHTML = html;

        // Add event listeners to controls and expand buttons
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

            // Expand button
            const expandBtn = card.querySelector('.param-card__expand');
            if (expandBtn) {
                expandBtn.addEventListener('click', () => {
                    this.toggleCardExpanded(key, card);
                });
            }
        });
    }

    /**
     * Toggle card expanded state
     */
    toggleCardExpanded(key, card) {
        if (this.expandedCards.has(key)) {
            this.expandedCards.delete(key);
            card.classList.remove('param-card--expanded');
        } else {
            this.expandedCards.add(key);
            card.classList.add('param-card--expanded');
        }
    }

    /**
     * Render a single parameter card with enhanced information
     * Note: Content is from controlled internal data
     */
    renderParameterCard(param, arch) {
        const value = this.parameterValues[arch.id][param.key];
        const controlHtml = this.renderControl(param, value);
        const isExpanded = this.expandedCards.has(param.key);

        // Build tradeoffs HTML
        let tradeoffsHtml = '';
        if (param.tradeoffs) {
            if (param.tradeoffs.increase && param.tradeoffs.decrease) {
                tradeoffsHtml = `
                    <div class="param-card__tradeoffs">
                        <div class="tradeoff tradeoff--increase">
                            <div class="tradeoff__label">↑ Increase</div>
                            <ul class="tradeoff__list">
                                ${param.tradeoffs.increase.map(t => `<li>${t}</li>`).join('')}
                            </ul>
                        </div>
                        <div class="tradeoff tradeoff--decrease">
                            <div class="tradeoff__label">↓ Decrease</div>
                            <ul class="tradeoff__list">
                                ${param.tradeoffs.decrease.map(t => `<li>${t}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                `;
            } else {
                // For dropdowns with option-specific tradeoffs
                tradeoffsHtml = `
                    <div class="param-card__options-info">
                        ${Object.entries(param.tradeoffs).map(([opt, traits]) => `
                            <div class="option-info ${opt === value ? 'option-info--active' : ''}">
                                <div class="option-info__name">${opt}</div>
                                <ul class="option-info__traits">
                                    ${Array.isArray(traits) ? traits.map(t => `<li>${t}</li>`).join('') : ''}
                                </ul>
                            </div>
                        `).join('')}
                    </div>
                `;
            }
        }

        // Build recommendations HTML
        let recommendationsHtml = '';
        if (param.recommendations) {
            recommendationsHtml = `
                <div class="param-card__recommendations">
                    <div class="recommendations__label">Recommendations</div>
                    <div class="recommendations__grid">
                        ${Object.entries(param.recommendations).map(([context, rec]) => `
                            <div class="recommendation">
                                <span class="recommendation__context">${context}:</span>
                                <span class="recommendation__value">${rec}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        return `
            <div class="param-card ${isExpanded ? 'param-card--expanded' : ''}" data-key="${param.key}">
                <div class="param-card__header">
                    <span class="param-card__name">
                        ${param.name}
                    </span>
                    <button class="param-card__expand" title="Show more details">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                            <path d="M4 6L8 10L12 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                        </svg>
                    </button>
                </div>

                <div class="param-card__description">
                    ${param.description || param.tip}
                </div>

                <div class="param-card__control">
                    ${controlHtml}
                </div>

                <div class="param-card__tip">
                    <div class="param-card__tip-icon">💡</div>
                    <div class="param-card__tip-text">${param.tip}</div>
                </div>

                <div class="param-card__expanded-content">
                    ${tradeoffsHtml}

                    ${param.research ? `
                        <div class="param-card__research">
                            <div class="research__label">Research Notes</div>
                            <div class="research__text">${param.research}</div>
                        </div>
                    ` : ''}

                    ${recommendationsHtml}
                </div>

                <div class="param-card__indicators">
                    <span class="param-card__indicator param-card__indicator--compute" data-level="${param.compute}">
                        Compute: ${param.compute}
                    </span>
                    <span class="param-card__indicator param-card__indicator--impact" data-level="${param.impact}">
                        Impact: ${param.impact}
                    </span>
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

        // Update active state for option-specific tradeoffs
        const optionInfos = card.querySelectorAll('.option-info');
        optionInfos.forEach(info => {
            const optName = info.querySelector('.option-info__name')?.textContent;
            info.classList.toggle('option-info--active', optName === value);
        });
    }

    /**
     * Filter parameters by search query
     */
    filterParameters(query) {
        const q = query.toLowerCase();
        this.elements.paramsGrid?.querySelectorAll('.param-card').forEach(card => {
            const name = card.querySelector('.param-card__name')?.textContent.toLowerCase() || '';
            const desc = card.querySelector('.param-card__description')?.textContent.toLowerCase() || '';
            const tip = card.querySelector('.param-card__tip-text')?.textContent.toLowerCase() || '';
            const research = card.querySelector('.research__text')?.textContent.toLowerCase() || '';
            const matches = name.includes(q) || desc.includes(q) || tip.includes(q) || research.includes(q);
            card.style.display = matches ? '' : 'none';
        });

        // Hide empty categories
        this.elements.paramsGrid?.querySelectorAll('.param-category').forEach(cat => {
            const visibleCards = cat.querySelectorAll('.param-card[style=""], .param-card:not([style])');
            cat.style.display = visibleCards.length > 0 ? '' : 'none';
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
            const originalText = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(() => {
                btn.textContent = originalText;
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
