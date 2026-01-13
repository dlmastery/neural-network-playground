# Design Spaces Tab Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a 6th tab to the Neural Network Playground providing an interactive educational reference for architecture design parameters across 10 modern deep learning architectures.

**Architecture:** Two-column layout with architecture navigator (left) and parameter cards (right). Data-driven design with all parameters defined in a central data file. Dynamically rendered UI with interactive controls.

**Tech Stack:** Vanilla JavaScript (ES6 modules), HTML5, CSS3 (glassmorphism, animations)

**Security Note:** All innerHTML usage in this plan uses trusted, static data defined in code (architecture definitions). No user input is rendered as HTML.

---

## Task 1: Add Tab Button and Panel Structure

**Files:**
- Modify: `index.html:58-64` (tab navigation)
- Modify: `index.html` (add new panel after diffusion-panel, around line 1230)

**Step 1: Add the design-spaces tab button**

In `index.html`, find the tab navigation section and add the new tab after diffusion:

```html
<button class="tab-btn" data-tab="design-spaces">Design Spaces</button>
```

**Step 2: Add the design-spaces panel skeleton**

Add after the diffusion-panel closing `</div>` (before the diffusion explainer modal):

```html
<!-- Design Spaces Tab Panel -->
<div class="tab-panel" id="design-spaces-panel">
    <!-- Controls Bar -->
    <div class="controls-bar glass-panel design-spaces-controls">
        <div class="control-section">
            <span class="control-section__label">Architecture</span>
            <span class="inline-metric__value" id="design-spaces-current-arch">Transformers</span>
        </div>

        <div class="control-divider"></div>

        <div class="control-section">
            <input type="text" class="input input--compact" id="design-spaces-search" placeholder="Search parameters...">
        </div>

        <div class="control-divider"></div>

        <div class="control-section">
            <span class="control-section__label">View</span>
            <div class="btn-group">
                <button class="btn btn--small btn--active" data-view="cards" id="design-spaces-view-cards">Cards</button>
                <button class="btn btn--small" data-view="table" id="design-spaces-view-table">Table</button>
            </div>
        </div>

        <div class="control-section">
            <button class="btn btn--accent" id="design-spaces-copy-btn">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg>
                Copy Config
            </button>
        </div>
    </div>

    <!-- Main Content -->
    <div class="main-content design-spaces-main">
        <!-- Left Sidebar - Architecture Navigator -->
        <aside class="sidebar sidebar--left glass-panel design-spaces-nav">
            <div class="sidebar__header">
                <h3>Architectures</h3>
            </div>
            <div class="sidebar__content" id="design-spaces-arch-list">
                <!-- Populated by JS -->
            </div>
        </aside>

        <!-- Center - Parameter Cards -->
        <section class="canvas-area design-spaces-params">
            <div class="design-spaces-params-header">
                <h2 id="design-spaces-arch-title">Transformers</h2>
                <p id="design-spaces-arch-desc">Attention-based architecture for sequence modeling</p>
            </div>
            <div class="design-spaces-params-grid" id="design-spaces-params-grid">
                <!-- Populated by JS -->
            </div>
        </section>

        <!-- Right Sidebar - Quick Reference -->
        <aside class="sidebar sidebar--right glass-panel design-spaces-info">
            <div class="sidebar__header">
                <h3>Quick Reference</h3>
            </div>
            <div class="sidebar__content" id="design-spaces-quick-ref">
                <div class="quick-ref-section">
                    <h4>When to Use</h4>
                    <p id="design-spaces-when-to-use">Sequential data, NLP, time series</p>
                </div>
                <div class="quick-ref-section">
                    <h4>Key Innovation</h4>
                    <p id="design-spaces-innovation">Self-attention mechanism</p>
                </div>
                <div class="quick-ref-section">
                    <h4>Typical Size</h4>
                    <p id="design-spaces-size">10M - 175B parameters</p>
                </div>
            </div>
        </aside>
    </div>
</div>
```

**Step 3: Verify tab switching works**

Open browser, click "Design Spaces" tab, verify empty panel appears.

**Step 4: Commit**

```bash
git add index.html
git commit -m "feat(design-spaces): add tab button and panel structure"
```

---

## Task 2: Create CSS Styles

**Files:**
- Create: `css/design-spaces.css`
- Modify: `index.html:20` (add stylesheet link)

**Step 1: Create the CSS file with base styles**

Create `css/design-spaces.css` with layout, parameter cards, architecture navigator, responsive breakpoints, toggle controls, and animations. See design document for full color scheme and visual specifications.

**Step 2: Add stylesheet link to index.html**

After the diffusion.css link, add:

```html
<link rel="stylesheet" href="css/design-spaces.css">
```

**Step 3: Verify styles apply**

Open browser, check Design Spaces tab has proper layout.

**Step 4: Commit**

```bash
git add css/design-spaces.css index.html
git commit -m "feat(design-spaces): add CSS styles"
```

---

## Task 3: Create Parameter Data File

**Files:**
- Create: `js/design-spaces/design-spaces-data.js`

**Step 1: Create the data file with all architecture parameters**

Create `js/design-spaces/design-spaces-data.js` with the complete `architectures` array containing all 10 architectures:
- Transformers (10 params)
- LLMs (10 params)
- Diffusion (10 params)
- GNNs (10 params)
- CNNs (10 params)
- RNNs/LSTMs (10 params)
- SSMs/Mamba (10 params)
- ViTs (10 params)
- VAEs (10 params)
- GANs (10 params)

Each parameter includes: name, key, type, options/range, default, tip, compute indicator, impact indicator.

**Step 2: Commit**

```bash
git add js/design-spaces/design-spaces-data.js
git commit -m "feat(design-spaces): add comprehensive parameter data for 10 architectures"
```

---

## Task 4: Create Main App Controller

**Files:**
- Create: `js/design-spaces/design-spaces-app.js`

**Step 1: Create the app controller**

Create `js/design-spaces/design-spaces-app.js` with DesignSpacesApp class containing:
- init() - Initialize tab
- cacheElements() - Cache DOM references
- setupEventListeners() - Search, view toggle, copy button
- renderArchitectureNav() - Render sidebar with 10 architectures
- selectArchitecture(archId) - Switch active architecture
- renderParameters(arch) - Render parameter cards grid
- renderParameterCard(param, arch) - Render individual card
- renderControl(param, value) - Render slider/dropdown/toggle
- updateParameterValue() - Handle parameter changes
- filterParameters(query) - Filter by search
- setView(mode) - Toggle cards/table view
- copyConfig() - Copy JSON to clipboard

**Step 2: Commit**

```bash
git add js/design-spaces/design-spaces-app.js
git commit -m "feat(design-spaces): add main app controller"
```

---

## Task 5: Integrate with Main App

**Files:**
- Modify: `js/main.js`

**Step 1: Find the tab switching logic in main.js**

Search for the tab button click handler and add initialization for design-spaces tab.

**Step 2: Add design-spaces import and initialization**

Add import at top:
```javascript
import { designSpacesApp } from './design-spaces/design-spaces-app.js';
```

Add to tab switch handler (find where `diffusionApp.init()` is called):
```javascript
case 'design-spaces':
    designSpacesApp.init();
    break;
```

**Step 3: Verify tab works**

Open browser, click Design Spaces tab, verify architectures and parameters load.

**Step 4: Commit**

```bash
git add js/main.js
git commit -m "feat(design-spaces): integrate with main app"
```

---

## Task 6: Final Integration Test

**Step 1: Test all architectures load correctly**

Open browser, navigate to Design Spaces tab, click through all 10 architectures.

**Step 2: Test parameter interactions**

- Adjust sliders and verify values update
- Change dropdowns and verify selection persists
- Toggle switches and verify state changes
- Use search to filter parameters
- Click Copy Config and verify JSON in clipboard

**Step 3: Test responsive design**

Resize browser to test breakpoints at 1199px, 899px, 599px.

**Step 4: Commit any fixes**

If any issues found, fix and commit.

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat(design-spaces): complete implementation with all 10 architectures"
```

---

## Summary

This implementation plan creates a fully functional Design Spaces tab with:

- **10 architectures** with ~10 parameters each (100+ total parameters)
- **Interactive controls** (sliders, dropdowns, toggles)
- **Best practice guidance** for each parameter
- **Search and filter** functionality
- **Copy Config** to export settings
- **Responsive design** for all screen sizes
- **Smooth animations** and visual polish

The tab serves as a comprehensive cheat-sheet for practitioners building deep learning models.
