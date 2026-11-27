"""
API Test Suite - Restructured & Operational
Tests all FastAPI endpoints aligned with app/main.py
"""
import requests
import os
import json

PORT = int(os.getenv("PORT", 8000))
BASE_URL = f"http://localhost:{PORT}"

def test_root():
    """Test root endpoint (serves the HTML dashboard)"""
    print("\n" + "="*70)
    print("TEST 1: Root Endpoint (Dashboard HTML)")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert "html" in response.headers.get("content-type", "").lower()
    assert "<!DOCTYPE html>" in response.text or "<html" in response.text
    print("✅ Dashboard UI successfully served as HTML.")

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*70)
    print("TEST 2: Health Check")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print(f"✅ Status: {data['status']}")
    print(f"✅ Components: {data['components']}")
    print(f"✅ Performance: {data['performance']}")

def test_list_investors():
    """Test listing investor profiles"""
    print("\n" + "="*70)
    print("TEST 3: List Investors")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/investors?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "investors" in data
    print(f"✅ Total investors available: {data['total_count']}")
    print(f"✅ Data source: {data['data_source']}")
    print(f"✅ Sample investor: {data['investors'][0]}")

def test_get_investor():
    """Test getting single investor details"""
    print("\n" + "="*70)
    print("TEST 4: Get Investor Details")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/investors/INV001")
    assert response.status_code == 200
    data = response.json()
    assert data["investor_id"] == "INV001"
    print(f"✅ Investor ID: {data['investor_id']}")
    print(f"✅ Profile: {data['profile']}")
    print(f"✅ Suitable products count: {data['suitable_products_count']}")

def test_list_products():
    """Test listing available investment products"""
    print("\n" + "="*70)
    print("TEST 5: List Products")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/products")
    assert response.status_code == 200
    data = response.json()
    assert "products" in data
    print(f"✅ Total products in catalog: {data['total_count']}")
    print(f"✅ Sample product: {data['products'][0]}")

def test_consult():
    """Test standard advisory RAG pipeline query"""
    print("\n" + "="*70)
    print("TEST 6: Advisory Consultation (RAG + Safety Rules)")
    print("="*70)
    
    payload = {
        "investor_id": "INV001",
        "query": "I want high-growth investments in tech sectors with moderate risk tolerance."
    }
    
    response = requests.post(f"{BASE_URL}/consult", json=payload)
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Status: {data['status']}")
    print(f"✅ Data Source: {data['data_source']}")
    if data['status'] == 'APPROVED':
        print(f"✅ Recommended products: {data['recommendation']['products']}")
        print(f"✅ Allocations: {data['recommendation']['allocations']}")
    else:
        print(f"⚠️  Rejected: {data['validation_summary']['violations']}")

def test_consult_invalid_investor():
    """Test advisory query for invalid investor profile ID"""
    print("\n" + "="*70)
    print("TEST 7: Advisory Consultation (Invalid Investor ID)")
    print("="*70)
    
    payload = {
        "investor_id": "INV_NOT_EXISTS",
        "query": "I want secure investment schemes."
    }
    
    response = requests.post(f"{BASE_URL}/consult", json=payload)
    assert response.status_code == 404
    print("✅ Correctly rejected with 404 for invalid investor.")

def test_audit_stats():
    """Test audit log statistical endpoint"""
    print("\n" + "="*70)
    print("TEST 8: Get Compliance Logging Statistics")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/audit/stats")
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Total audited requests: {data['total_requests']}")
    print(f"✅ Approved: {data['approved_count']}")
    print(f"✅ Rejected: {data['rejected_count']}")

def test_prometheus_metrics():
    """Test Prometheus metrics endpoint output"""
    print("\n" + "="*70)
    print("TEST 9: Prometheus Telemetry Scrape")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/metrics")
    assert response.status_code == 200
    assert "# HELP http_requests_total" in response.text
    assert "http_requests_total" in response.text
    print("✅ Prometheus metrics gathered successfully.")
    print("✅ Verified existence of 'http_requests_total' metric.")

def run_all_tests():
    """Execute all tests in order"""
    print("\n" + "="*70)
    print("🧪 RUNNING ETHICAL AI ADVISOR API TESTS")
    print("="*70)
    
    tests = [
        test_root,
        test_health,
        test_list_investors,
        test_get_investor,
        test_list_products,
        test_consult,
        test_consult_invalid_investor,
        test_audit_stats,
        test_prometheus_metrics
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            failed += 1
            
    print("\n" + "="*70)
    print("📊 API TEST SUMMARY")
    print("="*70)
    print(f"Total tests run: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    print("="*70)
    
    return failed == 0

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Starting API tests...")
    print(f"Targeting: {BASE_URL}")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not responding correctly.")
            exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it is running.")
        exit(1)
        
    success = run_all_tests()
    exit(0 if success else 1)
