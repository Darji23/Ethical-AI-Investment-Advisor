/**
 * Ethical AI Investment Advisor - Interactive Client Dashboard
 * Handles tab swapping, client profile filtering, RAG queries, live compliance audits,
 * and the product catalog.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Global State
    let allInvestors = [];
    let selectedInvestor = null;
    let productsCatalog = [];
    let totalInvestors = null;     // total profiles in MongoDB (from /investors)
    let activeRules = [];          // R1–R7 from /rules (config/suitability_rules.json)

    const escapeHtml = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

    // Backend status pill, wired to /health instead of a hard-coded "online"
    async function checkBackendHealth() {
        const dot = document.getElementById('backend-status-dot');
        const text = document.getElementById('backend-status-text');
        try {
            const res = await fetch('/health');
            if (!res.ok) throw new Error(res.status);
            const h = await res.json();
            const dbOk = h.components && h.components.mongodb_database === 'healthy';
            dot.className = `status-indicator ${dbOk ? 'online' : 'offline'}`;
            text.textContent = dbOk ? 'Local demo · backend online' : 'Local demo · database unavailable';
        } catch (e) {
            dot.className = 'status-indicator offline';
            text.textContent = 'Local demo · backend offline';
        }
    }

    async function loadRules() {
        try {
            const res = await fetch('/rules');
            if (!res.ok) throw new Error(res.status);
            activeRules = (await res.json()).rules || [];
        } catch (e) {
            console.error('Failed to load rules:', e);
            activeRules = [];
        }
    }
    
    // Tab Elements
    const tabs = {
        advisor: { btn: document.getElementById('tab-advisor-btn'), pane: document.getElementById('tab-advisor') },
        catalog: { btn: document.getElementById('tab-catalog-btn'), pane: document.getElementById('tab-catalog') }
    };
    
    // =========================================================================
    // 1. NAVIGATION TAB MANAGEMENT
    // =========================================================================
    function switchTab(targetTabKey) {
        Object.keys(tabs).forEach(key => {
            if (key === targetTabKey) {
                tabs[key].btn.classList.add('active');
                tabs[key].pane.classList.add('active');
            } else {
                tabs[key].btn.classList.remove('active');
                tabs[key].pane.classList.remove('active');
            }
        });
        
        // Trigger initial data loads if entering a tab
        if (targetTabKey === 'catalog' && productsCatalog.length === 0) {
            loadProductsCatalog();
        }
    }
    
    tabs.advisor.btn.addEventListener('click', () => switchTab('advisor'));
    tabs.catalog.btn.addEventListener('click', () => switchTab('catalog'));
    
    // =========================================================================
    // 2. ADVISOR CONSOLE TAB - INVESTORS LIST & SEARCH
    // =========================================================================
    const investorListContainer = document.getElementById('investor-list-container');
    const investorSearchInput = document.getElementById('investor-search');
    
    async function loadInvestorProfiles() {
        try {
            const response = await fetch('/investors?limit=100');
            if (!response.ok) throw new Error('Failed to fetch investor list');
            
            const data = await response.json();
            allInvestors = data.investors || [];
            totalInvestors = data.total_available || allInvestors.length;

            document.getElementById('investor-count').textContent =
                `Showing ${allInvestors.length.toLocaleString()} of ${totalInvestors.toLocaleString()}`;
            renderInvestorsList(allInvestors);
        } catch (error) {
            console.error('Error loading investors:', error);
            investorListContainer.innerHTML = `
                <div class="empty-select-message" style="color: var(--color-danger)">
                    <i class="fa-solid fa-triangle-exclamation"></i> Error loading client profiles.
                </div>`;
        }
    }
    
    function renderInvestorsList(investors, message) {
        if (investors.length === 0) {
            investorListContainer.innerHTML = `<p class="empty-select-message">${escapeHtml(message || 'No matching clients found.')}</p>`;
            return;
        }

        investorListContainer.innerHTML = '';
        investors.forEach(inv => {
            const card = document.createElement('div');
            card.className = `investor-card ${selectedInvestor && selectedInvestor.investor_id === inv.investor_id ? 'selected' : ''} ${inv._lookup ? 'lookup-result' : ''}`;
            card.dataset.id = inv.investor_id;

            card.innerHTML = `
                <div class="inv-header-row">
                    <span class="inv-id">${inv.investor_id}${inv._lookup ? '<span class="lookup-tag">by ID</span>' : ''}</span>
                    <span class="inv-exp-tag ${inv.experience}">${inv.experience}</span>
                </div>
                <div class="inv-details-row">
                    <span>Age ${inv.age}</span>
                    <span>Risk: ${inv.risk_tolerance}/5</span>
                    <span>Horizon: ${inv.time_horizon}y</span>
                </div>`;
            
            card.addEventListener('click', () => selectInvestor(inv.investor_id));
            investorListContainer.appendChild(card);
        });
    }
    
    // Filter the loaded profiles; a full ID (INV + 5 digits) outside them is looked up via /investors/{id}
    let lookupSeq = 0;
    investorSearchInput.addEventListener('input', async (e) => {
        const query = e.target.value.toLowerCase().trim();
        const seq = ++lookupSeq;
        if (!query) {
            renderInvestorsList(allInvestors);
            return;
        }
        const filtered = allInvestors.filter(inv =>
            inv.investor_id.toLowerCase().includes(query) ||
            inv.experience.toLowerCase().includes(query) ||
            inv.age.toString().includes(query)
        );
        const idQuery = query.toUpperCase();
        if (!/^INV\d{5}$/.test(idQuery) || filtered.some(inv => inv.investor_id === idQuery)) {
            renderInvestorsList(filtered);
            return;
        }
        renderInvestorsList([], `Looking up ${idQuery}…`);
        try {
            const res = await fetch(`/investors/${encodeURIComponent(idQuery)}`);
            if (seq !== lookupSeq) return;  // a newer keystroke superseded this lookup
            if (res.status === 404) {
                renderInvestorsList([], `No investor with ID ${idQuery}.`);
                return;
            }
            if (!res.ok) throw new Error(res.status);
            const d = await res.json();
            renderInvestorsList([{
                investor_id: d.investor_id,
                age: d.profile.age,
                risk_tolerance: d.profile.risk_tolerance,
                experience: d.profile.investment_experience,
                time_horizon: d.profile.time_horizon_years,
                _lookup: true
            }]);
        } catch (err) {
            if (seq !== lookupSeq) return;
            console.error('Investor lookup failed:', err);
            renderInvestorsList([], `Lookup for ${idQuery} failed.`);
        }
    });
    
    // Select an investor and show summary
    const activeInvestorSummary = document.getElementById('active-investor-summary');
    const queryBoxWrapper = document.getElementById('query-box-wrapper');
    const consultResultsArea = document.getElementById('consult-results');
    
    async function selectInvestor(investorId) {
        // Highlight active card
        document.querySelectorAll('.investor-card').forEach(card => {
            if (card.dataset.id === investorId) {
                card.classList.add('selected');
            } else {
                card.classList.remove('selected');
            }
        });
        
        activeInvestorSummary.innerHTML = `<div class="loading-spinner"><i class="fa-solid fa-circle-notch fa-spin"></i> Fetching profile...</div>`;
        queryBoxWrapper.style.display = 'none';
        consultResultsArea.style.display = 'none';
        resetCompliancePanel();
        
        try {
            const response = await fetch(`/investors/${investorId}`);
            if (!response.ok) throw new Error('Failed to fetch investor details');
            
            const data = await response.json();
            selectedInvestor = {
                investor_id: data.investor_id,
                ...data.profile
            };
            
            // Render detailed profile card
            renderActiveInvestorSummary(selectedInvestor);
            queryBoxWrapper.style.display = 'flex';
            
            // Auto-fill suggested query context
            document.getElementById('client-query').value = '';
        } catch (error) {
            console.error('Error fetching investor profile:', error);
            activeInvestorSummary.innerHTML = `<p class="empty-select-message" style="color: var(--color-danger)">Failed to load profile details.</p>`;
        }
    }
    
    function renderActiveInvestorSummary(profile) {
        activeInvestorSummary.innerHTML = `
            <div class="investor-profile-details-grid">
                <div class="profile-stat-box">
                    <span class="lbl">Client ID</span>
                    <span class="val">${profile.investor_id}</span>
                </div>
                <div class="profile-stat-box">
                    <span class="lbl">Age (R4)</span>
                    <span class="val">${profile.age} years</span>
                </div>
                <div class="profile-stat-box">
                    <span class="lbl">Experience (R5)</span>
                    <span class="val" style="text-transform: capitalize;">${profile.investment_experience}</span>
                </div>
                <div class="profile-stat-box">
                    <span class="lbl">Risk Tolerance (R3)</span>
                    <span class="val score-indicator-${profile.risk_tolerance}">${profile.risk_tolerance} / 5</span>
                </div>
                <div class="profile-stat-box">
                    <span class="lbl">Annual Income</span>
                    <span class="val">$${profile.annual_income.toLocaleString()}</span>
                </div>
                <div class="profile-stat-box">
                    <span class="lbl">Net Worth</span>
                    <span class="val">$${profile.net_worth.toLocaleString()}</span>
                </div>
                <div class="profile-stat-box">
                    <span class="lbl">Investable Assets (R7)</span>
                    <span class="val">$${(profile.investable_assets ?? 0).toLocaleString()}</span>
                </div>
                <div class="profile-stat-box">
                    <span class="lbl">Time Horizon (R2)</span>
                    <span class="val">${profile.time_horizon_years} years</span>
                </div>
                <div class="profile-stat-box">
                    <span class="lbl">Liquidity Needs (R2)</span>
                    <span class="val" style="text-transform: capitalize;">${escapeHtml(profile.liquidity_needs || '—')}</span>
                </div>
                <div class="profile-stat-box">
                    <span class="lbl">Objective (R2)</span>
                    <span class="val" style="text-transform: capitalize;">${escapeHtml(profile.investment_objective || '—')}</span>
                </div>
                ${(profile.profile_warnings && profile.profile_warnings.length) ? `
                <div class="profile-warnings">
                    <i class="fa-solid fa-triangle-exclamation"></i>
                    <strong>Profile warnings:</strong> ${profile.profile_warnings.map(escapeHtml).join('; ')}
                </div>` : ''}
            </div>`;
    }
    
    // Suggested query chips interaction
    document.querySelectorAll('.query-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const queryText = chip.textContent;
            const queryArea = document.getElementById('client-query');
            if (queryText === 'High-growth tech stocks') {
                queryArea.value = "I want aggressive growth stock options focused on technology sector ETFs.";
            } else if (queryText === 'Safe conservative income') {
                queryArea.value = "I need safe and stable income-generating assets to preserve my capital for retirement.";
            } else {
                queryArea.value = "I am looking for a balanced index fund split between US growth and stable corporate bonds.";
            }
        });
    });
    
    // =========================================================================
    // 3. CONSULTATION REQUESTS
    // =========================================================================
    const generateAdviceBtn = document.getElementById('generate-advice-btn');
    const clientQueryInput = document.getElementById('client-query');
    
    generateAdviceBtn.addEventListener('click', async () => {
        if (!selectedInvestor) return;
        const queryText = clientQueryInput.value.trim();
        if (queryText.length < 10) {
            alert('Please provide a descriptive query (minimum 10 characters).');
            return;
        }
        
        // UI Loading State
        setButtonLoading(generateAdviceBtn, true);
        consultResultsArea.style.display = 'none';
        
        try {
            const response = await fetch('/consult', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    investor_id: selectedInvestor.investor_id,
                    query: queryText
                })
            });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Internal server error');
            }
            
            const result = await response.json();
            renderConsultationResults(result);
        } catch (error) {
            console.error('Consultation failed:', error);
            alert(`Error: ${error.message}`);
        } finally {
            setButtonLoading(generateAdviceBtn, false);
        }
    });
    
    function setButtonLoading(button, isLoading) {
        const content = button.querySelector('.btn-content');
        const loading = button.querySelector('.btn-loading');
        
        if (isLoading) {
            button.disabled = true;
            content.style.display = 'none';
            loading.style.display = 'inline-block';
        } else {
            button.disabled = false;
            content.style.display = 'inline-block';
            loading.style.display = 'none';
        }
    }
    
    function renderConsultationResults(data) {
        consultResultsArea.style.display = 'flex';
        
        const outcomeBanner = document.getElementById('outcome-banner-element');
        const outcomeIcon = document.getElementById('outcome-icon-element');
        const outcomeTitle = document.getElementById('outcome-title');
        const outcomeDesc = document.getElementById('outcome-desc');
        
        // 1. Outcome Status Layout
        const outOfScope = data.status === 'OUT_OF_SCOPE';
        if (outOfScope) {
            outcomeBanner.className = 'outcome-banner out-of-scope';
            outcomeIcon.innerHTML = '<i class="fa-solid fa-circle-question"></i>';
            outcomeTitle.textContent = 'NO RECOMMENDATION';
            outcomeDesc.textContent = "This request isn't about investing, so the advisory pipeline was not run. Ask about a portfolio, risk, income or growth.";
        } else if (data.status === 'APPROVED') {
            outcomeBanner.className = 'outcome-banner approved';
            outcomeIcon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
            outcomeTitle.textContent = 'PORTFOLIO APPROVED';
            outcomeDesc.textContent = data.validation_summary.fallback_used
                ? 'The AI proposal failed validation, so a validated fallback portfolio built from rule-compliant products is shown. It passes rules R1–R7.'
                : 'The AI-generated portfolio passes all suitability rules R1–R7.';
        } else {
            outcomeBanner.className = 'outcome-banner rejected';
            outcomeIcon.innerHTML = '<i class="fa-solid fa-circle-xmark"></i>';
            outcomeTitle.textContent = 'PORTFOLIO REJECTED';
            outcomeDesc.textContent = 'Violations flagged during suitability checks. Proposal blocked for compliance.';
        }
        
        // 2. Rationale Text
        const rationaleText = document.getElementById('proposal-rationale-text');
        rationaleText.textContent = outOfScope
            ? `Input scope check: ${(data.scope && data.scope.reason) || 'not an investment request.'}`
            : (data.recommendation ? data.recommendation.rationale : data.validation_summary.violations.join(', '));
        
        // 3. Allocations Visualizer
        const barsContainer = document.getElementById('allocation-bars-container');
        barsContainer.innerHTML = '';
        
        if (data.recommendation && data.recommendation.products && data.recommendation.products.length > 0) {
            data.recommendation.products.forEach((ticker, idx) => {
                const allocation = data.recommendation.allocations[idx];
                const barItem = document.createElement('div');
                barItem.className = 'alloc-bar-item';
                
                barItem.innerHTML = `
                    <div class="alloc-meta">
                        <span class="alloc-ticker">${ticker}</span>
                        <span class="alloc-val">${allocation}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: ${allocation}%;"></div>
                    </div>`;
                barsContainer.appendChild(barItem);
            });
        } else {
            barsContainer.innerHTML = `<p class="empty-select-message" style="padding: 1rem 0;">${outOfScope ? 'No portfolio: request was out of scope.' : 'No products allocated.'}</p>`;
        }
        
        // 4. Performance Latencies
        document.getElementById('metric-source').textContent = data.data_source.toUpperCase().replace('_', ' ');
        document.getElementById('metric-rag-time').textContent = `${data.performance_metrics.rag_pipeline_ms} ms`;
        document.getElementById('metric-total-time').textContent = `${data.performance_metrics.total_time_ms} ms`;
        
        // 5. Compliance Sidebar log
        updateCompliancePanel(
            data.validation_summary,
            !!(data.recommendation && data.recommendation.products && data.recommendation.products.length),
            outOfScope ? 'Not evaluated: request was out of scope, pipeline not run.' : null
        );
    }
    
    function resetCompliancePanel() {
        document.getElementById('compliance-risk-score').textContent = '0.00 / 5.0';
        document.getElementById('compliance-violation-count').textContent = '0';
        document.getElementById('compliance-violation-count').className = 'val count-neutral';
        document.getElementById('guardrail-checklist-container').innerHTML = `
            <p class="empty-select-message">No compliance checks executed yet. Initiate consultation to see log.</p>`;
    }
    
    function updateCompliancePanel(summary, hasPortfolio, skipReason) {
        const riskVal = document.getElementById('compliance-risk-score');
        const violationsVal = document.getElementById('compliance-violation-count');

        riskVal.textContent = `${summary.portfolio_risk.toFixed(2)} / 5.0`;
        violationsVal.textContent = summary.violations.length;
        violationsVal.className = summary.violations.length > 0 ? 'val count-violations' : 'val count-neutral';

        const listContainer = document.getElementById('guardrail-checklist-container');
        listContainer.innerHTML = '';

        const addItem = (state, title, detailHtml) => {
            const icon = { passed: 'fa-circle-check', failed: 'fa-circle-xmark', skipped: 'fa-circle-minus' }[state];
            const item = document.createElement('div');
            item.className = `checklist-item ${state}`;
            item.innerHTML = `
                <div class="check-icon"><i class="fa-solid ${icon}"></i></div>
                <div class="check-content">
                    <span class="check-title">${title}</span>
                    <span class="check-detail">${detailHtml}</span>
                </div>`;
            listContainer.appendChild(item);
        };

        // One row per rule in config/suitability_rules.json; the validator prefixes violations with "R1:" … "R7:"
        const matched = new Set();
        activeRules.forEach(rule => {
            const prefix = `${rule.short_id}:`;
            const hits = summary.violations.filter(v => v.startsWith(prefix));
            hits.forEach(v => matched.add(v));
            const title = `${escapeHtml(rule.short_id)} · ${escapeHtml(rule.name)}`;
            if (hits.length) {
                addItem('failed', title, hits.map(v => `<strong style="color: var(--color-danger);">${escapeHtml(v)}</strong>`).join('<br>'));
            } else if (!hasPortfolio) {
                addItem('skipped', title, skipReason || 'Not evaluated: no portfolio was proposed.');
            } else {
                addItem('passed', title, escapeHtml(rule.firm_policy_threshold));
            }
        });

        // Anything without a rule prefix (e.g. a referral or rejection reason) is still shown, never hidden
        summary.violations.filter(v => !matched.has(v)).forEach(v => {
            addItem('failed', 'Other finding', `<strong style="color: var(--color-danger);">${escapeHtml(v)}</strong>`);
        });

        // Append compliance notes as info blocks
        summary.compliance_notes.forEach(note => {
            const infoItem = document.createElement('div');
            infoItem.className = 'checklist-item passed';
            infoItem.style.borderLeft = '3px solid var(--accent-secondary)';
            
            infoItem.innerHTML = `
                <div class="check-icon" style="color: var(--accent-secondary);">
                    <i class="fa-solid fa-circle-info"></i>
                </div>
                <div class="check-content">
                    <span class="check-title">Compliance Disclosure Note</span>
                    <span class="check-detail">${escapeHtml(note)}</span>
                </div>`;
            listContainer.appendChild(infoItem);
        });
    }
    
    // =========================================================================
    // 4. PRODUCT CATALOG TAB
    // =========================================================================
    const catalogProductsContainer = document.getElementById('catalog-products-container');
    const filterClassSelect = document.getElementById('filter-class');
    const filterRiskSelect = document.getElementById('filter-risk');
    
    async function loadProductsCatalog() {
        try {
            const response = await fetch('/products');
            if (!response.ok) throw new Error('Product fetch failure');
            
            const data = await response.json();
            productsCatalog = data.products || [];
            const countEl = document.getElementById('catalog-count');
            if (countEl) countEl.textContent = productsCatalog.length.toLocaleString();
            renderProductsCatalog(productsCatalog);
        } catch (error) {
            console.error('Catalog error:', error);
            catalogProductsContainer.innerHTML = `<p class="empty-select-message" style="color: var(--color-danger)">Error loading product catalog.</p>`;
        }
    }
    
    function renderProductsCatalog(products) {
        catalogProductsContainer.innerHTML = '';
        
        if (products.length === 0) {
            catalogProductsContainer.innerHTML = '<p class="empty-select-message">No products match current filters.</p>';
            return;
        }
        
        products.forEach(p => {
            const card = document.createElement('div');
            card.className = 'product-card';
            
            card.innerHTML = `
                <div class="prod-header">
                    <span class="prod-ticker">${p.ticker}</span>
                    <span class="prod-tags">
                        ${(p.is_leveraged || p.is_inverse) ? `<span class="prod-lev-tag" title="Excluded from all portfolios by rule R1">${p.is_inverse ? 'Inverse' : 'Leveraged'} ${Math.abs(p.leverage_factor || 1)}x</span>` : ''}
                        <span class="prod-class-tag">${p.asset_class.replace('_', ' ')}</span>
                    </span>
                </div>
                <div class="prod-name">${p.name}</div>
                <div class="prod-details">
                    <div class="detail-block">
                        <span class="lbl">Risk Factor</span>
                        <span class="val risk score-indicator-${p.risk_score}">${p.risk_score} / 5</span>
                    </div>
                    <div class="detail-block">
                        <span class="lbl">Share Price</span>
                        <span class="val">$${Number(p.share_price || p.min_investment || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                    </div>
                </div>`;
            
            catalogProductsContainer.appendChild(card);
        });
    }
    
    // Trigger catalog filters
    function applyCatalogFilters() {
        const assetFilter = filterClassSelect.value;
        const riskFilter = parseInt(filterRiskSelect.value);
        
        let filtered = productsCatalog;
        
        if (assetFilter !== 'all') {
            filtered = filtered.filter(p => p.asset_class === assetFilter);
        }
        
        if (riskFilter > 0) {
            filtered = filtered.filter(p => p.risk_score <= riskFilter);
        }
        
        renderProductsCatalog(filtered);
    }
    
    filterClassSelect.addEventListener('change', applyCatalogFilters);
    filterRiskSelect.addEventListener('change', applyCatalogFilters);
    
    // =========================================================================
    // INITIALIZATION RUNS
    // =========================================================================
    checkBackendHealth();
    loadRules();
    loadInvestorProfiles();
});
