-- Database initialization script for Ethical AI Advisor
-- Phase 3: PostgreSQL Schema

-- Create investor_profiles table
CREATE TABLE IF NOT EXISTS investor_profiles (
    investor_id VARCHAR(50) PRIMARY KEY,
    age INTEGER NOT NULL CHECK (age >= 18 AND age <= 100),
    risk_tolerance INTEGER NOT NULL CHECK (risk_tolerance >= 1 AND risk_tolerance <= 5),
    annual_income DECIMAL(15, 2) NOT NULL CHECK (annual_income >= 0),
    net_worth DECIMAL(15, 2) NOT NULL CHECK (net_worth >= 0),
    investment_experience VARCHAR(20) NOT NULL CHECK (investment_experience IN ('beginner', 'intermediate', 'advanced')),
    time_horizon_years INTEGER NOT NULL CHECK (time_horizon_years >= 1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create products table
CREATE TABLE IF NOT EXISTS products (
    ticker VARCHAR(20) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    risk_score INTEGER NOT NULL CHECK (risk_score >= 1 AND risk_score <= 5),
    asset_class VARCHAR(50) NOT NULL CHECK (asset_class IN ('equity', 'fixed_income', 'alternative', 'cash', 'real_estate')),
    min_investment DECIMAL(15, 2) NOT NULL CHECK (min_investment >= 0),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create request_cache table (for Redis fallback)
CREATE TABLE IF NOT EXISTS request_cache (
    cache_key VARCHAR(255) PRIMARY KEY,
    investor_id VARCHAR(50) NOT NULL,
    query_hash VARCHAR(64) NOT NULL,
    response_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

-- Create audit_summary table (aggregated stats)
CREATE TABLE IF NOT EXISTS audit_summary (
    summary_date DATE PRIMARY KEY,
    total_requests INTEGER DEFAULT 0,
    approved_count INTEGER DEFAULT 0,
    rejected_count INTEGER DEFAULT 0,
    avg_portfolio_risk DECIMAL(3, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_investor_profiles_age ON investor_profiles(age);
CREATE INDEX IF NOT EXISTS idx_investor_profiles_risk ON investor_profiles(risk_tolerance);
CREATE INDEX IF NOT EXISTS idx_investor_profiles_experience ON investor_profiles(investment_experience);
CREATE INDEX IF NOT EXISTS idx_products_risk ON products(risk_score);
CREATE INDEX IF NOT EXISTS idx_products_asset_class ON products(asset_class);
CREATE INDEX IF NOT EXISTS idx_products_active ON products(is_active);
CREATE INDEX IF NOT EXISTS idx_request_cache_investor ON request_cache(investor_id);
CREATE INDEX IF NOT EXISTS idx_request_cache_expires ON request_cache(expires_at);

-- Insert sample data (first 10 investors from Phase 1)
-- Note: In production, all 100 profiles would be loaded programmatically
INSERT INTO investor_profiles (investor_id, age, risk_tolerance, annual_income, net_worth, investment_experience, time_horizon_years)
VALUES 
    ('INV001', 28, 5, 85000.00, 150000.00, 'intermediate', 30),
    ('INV002', 45, 3, 120000.00, 500000.00, 'advanced', 20),
    ('INV003', 62, 2, 90000.00, 1200000.00, 'advanced', 10),
    ('INV004', 24, 3, 55000.00, 25000.00, 'beginner', 35),
    ('INV005', 70, 1, 60000.00, 800000.00, 'intermediate', 15),
    ('INV006', 35, 4, 95000.00, 200000.00, 'intermediate', 25),
    ('INV007', 52, 2, 110000.00, 750000.00, 'advanced', 15),
    ('INV008', 29, 4, 78000.00, 120000.00, 'beginner', 30),
    ('INV009', 68, 1, 75000.00, 950000.00, 'advanced', 12),
    ('INV010', 41, 3, 130000.00, 400000.00, 'intermediate', 20)
ON CONFLICT (investor_id) DO NOTHING;

-- Insert all 25 products from Phase 1
INSERT INTO products (ticker, name, risk_score, asset_class, min_investment)
VALUES 
    -- Conservative (Risk 1)
    ('GOVT', 'US Treasury Bond Fund', 1, 'fixed_income', 1000.00),
    ('SHV', 'Short-Term Treasury ETF', 1, 'fixed_income', 500.00),
    ('VMFXX', 'Money Market Fund', 1, 'fixed_income', 3000.00),
    ('VTIP', 'Treasury Inflation-Protected Securities', 1, 'fixed_income', 1000.00),
    ('BND', 'Total Bond Market ETF', 1, 'fixed_income', 1000.00),
    
    -- Moderate Conservative (Risk 2)
    ('VBTLX', 'Vanguard Total Bond Market', 2, 'fixed_income', 3000.00),
    ('AGG', 'Core Bond ETF', 2, 'fixed_income', 1000.00),
    ('LQD', 'Investment Grade Corporate Bonds', 2, 'fixed_income', 2000.00),
    ('MUB', 'Municipal Bond ETF', 2, 'fixed_income', 1500.00),
    ('SCHP', 'TIPS ETF', 2, 'fixed_income', 1000.00),
    
    -- Moderate (Risk 3)
    ('VTI', 'Vanguard Total Stock Market', 3, 'equity', 1000.00),
    ('VWELX', 'Vanguard Wellington Fund', 3, 'equity', 3000.00),
    ('SPY', 'S&P 500 ETF', 3, 'equity', 500.00),
    ('VNQ', 'Real Estate Investment Trust ETF', 3, 'real_estate', 2000.00),
    ('VXUS', 'Total International Stock ETF', 3, 'equity', 1000.00),
    ('VBIAX', 'Balanced Index Fund', 3, 'equity', 3000.00),
    ('DIA', 'Dow Jones Industrial Average ETF', 3, 'equity', 1000.00),
    
    -- Moderate Aggressive (Risk 4)
    ('ARKK', 'ARK Innovation ETF', 4, 'equity', 500.00),
    ('QQQ', 'Invesco QQQ Trust', 4, 'equity', 1000.00),
    ('VGT', 'Technology Sector ETF', 4, 'equity', 1500.00),
    ('SOXX', 'Semiconductor ETF', 4, 'equity', 2000.00),
    ('VOOG', 'S&P 500 Growth ETF', 4, 'equity', 1000.00),
    
    -- Aggressive (Risk 5)
    ('BITW', 'Crypto Thematic ETF', 5, 'alternative', 5000.00),
    ('ARKW', 'Next Generation Internet ETF', 5, 'equity', 3000.00),
    ('SPXL', '3x Leveraged S&P 500 ETF', 5, 'alternative', 10000.00)
ON CONFLICT (ticker) DO NOTHING;

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for investor_profiles
CREATE TRIGGER update_investor_profiles_updated_at
    BEFORE UPDATE ON investor_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (for application user)
GRANT SELECT, INSERT, UPDATE ON investor_profiles TO advisor;
GRANT SELECT ON products TO advisor;
GRANT SELECT, INSERT, UPDATE, DELETE ON request_cache TO advisor;
GRANT SELECT, INSERT, UPDATE ON audit_summary TO advisor;

-- Create view for investor statistics
CREATE OR REPLACE VIEW investor_stats AS
SELECT 
    risk_tolerance,
    investment_experience,
    COUNT(*) as count,
    AVG(age) as avg_age,
    AVG(net_worth) as avg_net_worth
FROM investor_profiles
GROUP BY risk_tolerance, investment_experience;

GRANT SELECT ON investor_stats TO advisor;
