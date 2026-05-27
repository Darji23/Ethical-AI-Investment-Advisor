/**
 * Ethical AI Investment Advisor - Interactive Client Dashboard
 * Handles tab swapping, client profile filtering, RAG queries, live compliance audits,
 * and Fairness Audit simulation visualizations.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Global State
    let allInvestors = [];
    let selectedInvestor = null;
    let productsCatalog = [];
    
    // Tab Elements
    const tabs = {
        advisor: { btn: document.getElementById('tab-advisor-btn'), pane: document.getElementById('tab-advisor') },
        audit: { btn: document.getElementById('tab-audit-btn'), pane: document.getElementById('tab-audit') },
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
    tabs.audit.btn.addEventListener('click', () => switchTab('audit'));
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
            
            document.getElementById('investor-count').textContent = `${allInvestors.length} Profiles`;
            renderInvestorsList(allInvestors);
        } catch (error) {
            console.error('Error loading investors:', error);
            investorListContainer.innerHTML = `
                <div class="empty-select-message" style="color: var(--color-danger)">
                    <i class="fa-solid fa-triangle-exclamation"></i> Error loading client profiles.
                </div>`;
        }
    }
    
    function renderInvestorsList(investors) {
        if (investors.length === 0) {
            investorListContainer.innerHTML = '<p class="empty-select-message">No matching clients found.</p>';
            return;
        }
        
        investorListContainer.innerHTML = '';
        investors.forEach(inv => {
            const card = document.createElement('div');
            card.className = `investor-card ${selectedInvestor && selectedInvestor.investor_id === inv.investor_id ? 'selected' : ''}`;
            card.dataset.id = inv.investor_id;
            
            card.innerHTML = `
                <div class="inv-header-row">
                    <span class="inv-id">${inv.investor_id}</span>
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
    
    // Filter investors based on search query
    investorSearchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        if (!query) {
            renderInvestorsList(allInvestors);
            return;
        }
        const filtered = allInvestors.filter(inv => 
            inv.investor_id.toLowerCase().includes(query) ||
            inv.experience.toLowerCase().includes(query) ||
            inv.age.toString().includes(query)
        );
        renderInvestorsList(filtered);
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
                    <span class="lbl">Age</span>
                    <span class="val">${profile.age} years</span>
                </div>
                <div class="profile-stat-box">
                    <span class="lbl">Experience</span>
                    <span class="val" style="text-transform: capitalize;">${profile.investment_experience}</span>
                </div>
                <div class="profile-stat-box">
                    <span class="lbl">Risk Tolerance</span>
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
                    <span class="lbl">Time Horizon</span>
                    <span class="val">${profile.time_horizon_years} years</span>
                </div>
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
        if (data.status === 'APPROVED') {
            outcomeBanner.className = 'outcome-banner approved';
            outcomeIcon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
            outcomeTitle.textContent = 'PORTFOLIO APPROVED';
            outcomeDesc.textContent = 'The generated portfolio is fully compliant with all suitability guardrails.';
        } else {
            outcomeBanner.className = 'outcome-banner rejected';
            outcomeIcon.innerHTML = '<i class="fa-solid fa-circle-xmark"></i>';
            outcomeTitle.textContent = 'PORTFOLIO REJECTED';
            outcomeDesc.textContent = 'Violations flagged during suitability checks. Proposal blocked for compliance.';
        }
        
        // 2. Rationale Text
        const rationaleText = document.getElementById('proposal-rationale-text');
        rationaleText.textContent = data.recommendation ? data.recommendation.rationale : data.validation_summary.violations.join(', ');
        
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
            barsContainer.innerHTML = '<p class="empty-select-message" style="padding: 1rem 0;">No products allocated.</p>';
        }
        
        // 4. Performance Latencies
        document.getElementById('metric-source').textContent = data.data_source.toUpperCase().replace('_', ' ');
        document.getElementById('metric-rag-time').textContent = `${data.performance_metrics.rag_pipeline_ms} ms`;
        document.getElementById('metric-total-time').textContent = `${data.performance_metrics.total_time_ms} ms`;
        
        // 5. Compliance Sidebar log
        updateCompliancePanel(data.validation_summary);
    }
    
    function resetCompliancePanel() {
        document.getElementById('compliance-risk-score').textContent = '0.00 / 5.0';
        document.getElementById('compliance-violation-count').textContent = '0';
        document.getElementById('compliance-violation-count').className = 'val count-neutral';
        document.getElementById('guardrail-checklist-container').innerHTML = `
            <p class="empty-select-message">No compliance checks executed yet. Initiate consultation to see log.</p>`;
    }
    
    function updateCompliancePanel(summary) {
        const riskVal = document.getElementById('compliance-risk-score');
        const violationsVal = document.getElementById('compliance-violation-count');
        
        riskVal.textContent = `${summary.portfolio_risk.toFixed(2)} / 5.0`;
        violationsVal.textContent = summary.violations.length;
        
        if (summary.violations.length > 0) {
            violationsVal.className = 'val count-violations';
        } else {
            violationsVal.className = 'val count-neutral';
        }
        
        // Render checklist compliance
        const listContainer = document.getElementById('guardrail-checklist-container');
        listContainer.innerHTML = '';
        
        // 12 Safety Rules representation
        const complianceRules = [
            { id: 'approved_list', name: 'Product Master Registry Compliance', desc: 'Verify all recommended ticker symbols exist within the regulatory approved master catalog.' },
            { id: 'risk_tolerance', name: 'Maximum Risk Bound Check', desc: 'Ensure no single allocated security risk score exceeds the client\'s risk tolerance tier (+1 leeway maximum).' },
            { id: 'age_conservative', name: 'Senior Demographics Safeguard', desc: 'Enforce capital preservation by restricting high-risk asset allocations (risk score >= 4) for clients aged 65 or older.' },
            { id: 'beginner_complex', name: 'Novice Investor Defense', desc: 'Block complex, synthetic, leveraged, or crypto thematic assets for clients with beginner experience levels.' },
            { id: 'min_capital', name: 'Minimum Capital Suitability', desc: 'Confirm proposed allocations meet minimum starting capital parameters based on 10% of net worth.' },
            { id: 'diversification', name: 'Portfolio Diversification Check', desc: 'Require a minimum allocation across at least 2 distinct products, with no single asset exceeding 60% concentration.' },
            { id: 'experience_education', name: 'Investor Qualification Match', desc: 'Validate that the complexity of assets matches the investment history class of the client.' }
        ];
        
        complianceRules.forEach(rule => {
            // Check if this rule is violated based on text matching in violations list
            let isViolated = false;
            let violationText = '';
            
            summary.violations.forEach(v => {
                if (rule.id === 'approved_list' && v.includes('approved list')) {
                    isViolated = true;
                    violationText = v;
                } else if (rule.id === 'risk_tolerance' && (v.includes('exceeds tolerance') || v.includes('exceeded'))) {
                    isViolated = true;
                    violationText = v;
                } else if (rule.id === 'age_conservative' && v.includes('age 65') || v.includes('age 70+')) {
                    isViolated = true;
                    violationText = v;
                } else if (rule.id === 'beginner_complex' && (v.includes('Beginner') || v.includes('complex'))) {
                    isViolated = true;
                    violationText = v;
                } else if (rule.id === 'min_capital' && v.includes('minimum')) {
                    isViolated = true;
                    violationText = v;
                } else if (rule.id === 'diversification' && (v.includes('diversification') || v.includes('concentration'))) {
                    isViolated = true;
                    violationText = v;
                } else if (rule.id === 'experience_education' && v.includes('experience')) {
                    isViolated = true;
                    violationText = v;
                }
            });
            
            const checkItem = document.createElement('div');
            checkItem.className = `checklist-item ${isViolated ? 'failed' : 'passed'}`;
            
            checkItem.innerHTML = `
                <div class="check-icon">
                    <i class="fa-solid ${isViolated ? 'fa-circle-xmark' : 'fa-circle-check'}"></i>
                </div>
                <div class="check-content">
                    <span class="check-title">${rule.name}</span>
                    <span class="check-detail">${isViolated ? `<strong style="color: var(--color-danger);">${violationText}</strong>` : rule.desc}</span>
                </div>`;
            listContainer.appendChild(checkItem);
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
                    <span class="check-detail">${note}</span>
                </div>`;
            listContainer.appendChild(infoItem);
        });
    }
    
    // =========================================================================
    // 4. FAIRNESS AUDIT TAB
    // =========================================================================
    const runAuditBtn = document.getElementById('run-audit-btn');
    const auditResultsGrid = document.getElementById('audit-results-grid-wrapper');
    const auditEmptyView = document.getElementById('audit-empty-view');
    const auditTableBody = document.getElementById('audit-table-body');
    
    runAuditBtn.addEventListener('click', async () => {
        setButtonLoading(runAuditBtn, true);
        auditResultsGrid.style.display = 'none';
        auditEmptyView.style.display = 'none';
        
        try {
            const response = await fetch('/audit/simulate', { method: 'POST' });
            if (!response.ok) throw new Error('Simulation calculation failure');
            
            const data = await response.json();
            renderFairnessAudit(data);
        } catch (error) {
            console.error('Audit failed:', error);
            alert('Failed to execute simulation fairness metrics.');
            auditEmptyView.style.display = 'flex';
        } finally {
            setButtonLoading(runAuditBtn, false);
        }
    });
    
    function renderFairnessAudit(data) {
        auditResultsGrid.style.display = 'grid';
        
        // 1. Verdict Summary Cards
        const scoreVal = document.getElementById('audit-score-val');
        const verdictDesc = document.getElementById('audit-verdict-desc');
        const verdictBox = document.getElementById('audit-verdict-box');
        
        const score = data.overall_fairness.fairness_score;
        scoreVal.textContent = `Fairness Score: ${score.toFixed(1)} / 100`;
        
        if (data.overall_fairness.overall_pass) {
            verdictBox.className = 'audit-verdict-box approved';
            verdictBox.querySelector('i').className = 'fa-solid fa-circle-check verdict-icon text-success';
            verdictDesc.innerHTML = `<strong>PASS:</strong> System achieves demographic parity (<20% gap) across all audited dimensions.`;
        } else {
            verdictBox.className = 'audit-verdict-box rejected';
            verdictBox.querySelector('i').className = 'fa-solid fa-circle-xmark verdict-icon text-danger';
            verdictDesc.innerHTML = `<strong>WARNING:</strong> System fails parity checks. Structural disparities found.`;
        }
        
        // 2. Parity Gap Progress bars animations
        updateParityProgressBar('audit-gap-age', data.by_age._summary.demographic_parity_gap);
        updateParityProgressBar('audit-gap-exp', data.by_experience._summary.demographic_parity_gap);
        updateParityProgressBar('audit-gap-income', data.by_income._summary.demographic_parity_gap);
        
        // 3. Render granular table breakdown
        auditTableBody.innerHTML = '';
        
        renderTableDimensionGroup('Age Brackets', data.by_age);
        renderTableDimensionGroup('Experience Levels', data.by_experience);
        renderTableDimensionGroup('Income Tiers', data.by_income);
    }
    
    function updateParityProgressBar(elementId, gapFraction) {
        const bar = document.getElementById(elementId);
        const percentage = Math.round(gapFraction * 100);
        bar.style.width = `${percentage}%`;
        bar.textContent = `${percentage}%`;
        
        // Dynamic colors
        if (percentage < 12) {
            bar.className = 'progress-bar bg-success';
        } else if (percentage < 20) {
            bar.className = 'progress-bar bg-warning';
        } else {
            bar.className = 'progress-bar bg-danger';
        }
    }
    
    function renderTableDimensionGroup(groupName, dimensionData) {
        // Group heading row
        const headingRow = document.createElement('tr');
        headingRow.innerHTML = `<td colspan="5" class="table-group-header">${groupName}</td>`;
        auditTableBody.appendChild(headingRow);
        
        // Iterate groups within dimension
        Object.keys(dimensionData).forEach(key => {
            if (key.startsWith('_')) return; // skip summary keys
            
            const metrics = dimensionData[key];
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td style="font-weight: 600; color: #ffffff; text-transform: capitalize;">${key.replace('_', ' ')}</td>
                <td>${metrics.total_tests}</td>
                <td><strong style="color: ${metrics.approval_rate > 0.8 ? 'var(--color-success)' : 'var(--text-primary)'};">${Math.round(metrics.approval_rate * 100)}%</strong></td>
                <td>${metrics.avg_violations.toFixed(2)}</td>
                <td>${metrics.avg_portfolio_risk.toFixed(2)} / 5.0</td>`;
            
            auditTableBody.appendChild(row);
        });
    }
    
    // =========================================================================
    // 5. PRODUCT CATALOG TAB
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
                    <span class="prod-class-tag">${p.asset_class.replace('_', ' ')}</span>
                </div>
                <div class="prod-name">${p.name}</div>
                <div class="prod-details">
                    <div class="detail-block">
                        <span class="lbl">Risk Factor</span>
                        <span class="val risk score-indicator-${p.risk_score}">${p.risk_score} / 5</span>
                    </div>
                    <div class="detail-block">
                        <span class="lbl">Min Capital</span>
                        <span class="val">$${p.min_investment.toLocaleString()}</span>
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
    loadInvestorProfiles();
});
