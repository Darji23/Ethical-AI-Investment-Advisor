"""
API Test Suite for Phase 3
Tests all FastAPI endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint"""
    print("\n" + "="*70)
    print("TEST 1: Root Endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    
    data = response.json()
    print(f"✅ Service: {data['service']}")
    print(f"✅ Version: {data['version']}")
    print(f"✅ Status: {data['status']}")
    print(f"✅ AI Models: {data['ai_models']}")

def test_health():
    """Test health check"""
    print("\n" + "="*70)
    print("TEST 2: Health Check")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    
    data = response.json()
    print(f"✅ Status: {data['status']}")
    print(f"✅ Components: {data['components']}")

def test_list_investors():
    """Test list investors"""
    print("\n" + "="*70)
    print("TEST 3: List Investors")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/investors?limit=5")
    assert response.status_code == 200
    
    data = response.json()
    print(f"✅ Total investors: {data['total_count']}")
    print(f"✅ First investor: {data['investors'][0]['investor_id']}")

def test_get_investor():
    """Test get specific investor"""
    print("\n" + "="*70)
    print("TEST 4: Get Investor")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/investors/INV001")
    assert response.status_code == 200
    
    data = response.json()
    print(f"✅ Investor ID: {data['investor_id']}")
    print(f"✅ Age: {data['profile']['age']}")
    print(f"✅ Risk Tolerance: {data['profile']['risk_tolerance']}")
    print(f"✅ Suitable Products: {data['suitable_products_count']}")

def test_list_products():
    """Test list products"""
    print("\n" + "="*70)
    print("TEST 5: List Products")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/products")
    assert response.status_code == 200
    
    data = response.json()
    print(f"✅ Total products: {data['total_count']}")
    print(f"✅ First product: {data['products'][0]['ticker']}")

def test_get_product():
    """Test get specific product"""
    print("\n" + "="*70)
    print("TEST 6: Get Product")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/products/VTI")
    assert response.status_code == 200
    
    data = response.json()
    print(f"✅ Ticker: {data['ticker']}")
    print(f"✅ Name: {data['name']}")
    print(f"✅ Risk Score: {data['risk_score']}")

def test_consult_young_aggressive():
    """Test consultation for young aggressive investor"""
    print("\n" + "="*70)
    print("TEST 7: Consult - Young Aggressive Investor")
    print("="*70)
    
    payload = {
        "investor_id": "INV001",
        "query": "I want aggressive growth with tech exposure"
    }
    
    response = requests.post(f"{BASE_URL}/consult", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    print(f"✅ Status: {data['status']}")
    print(f"✅ Investor: {data['investor_summary']}")
    
    if data['status'] == 'APPROVED':
        print(f"✅ Products: {data['recommendation']['products']}")
        print(f"✅ Allocations: {data['recommendation']['allocations']}")
        print(f"✅ Rationale: {data['recommendation']['rationale'][:100]}...")
    else:
        print(f"⚠️  Rejected: {data['validation_summary']['violations']}")

def test_consult_elderly_conservative():
    """Test consultation for elderly conservative investor"""
    print("\n" + "="*70)
    print("TEST 8: Consult - Elderly Conservative Investor")
    print("="*70)
    
    payload = {
        "investor_id": "INV095",  # Elderly investor
        "query": "I need safe stable income for retirement"
    }
    
    response = requests.post(f"{BASE_URL}/consult", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    print(f"✅ Status: {data['status']}")
    print(f"✅ Investor: {data['investor_summary']}")
    
    if data['status'] == 'APPROVED':
        print(f"✅ Products: {data['recommendation']['products']}")
        print(f"✅ Allocations: {data['recommendation']['allocations']}")
        print(f"✅ Portfolio Risk: {data['validation_summary']['portfolio_risk']}")

def test_consult_invalid_investor():
    """Test consultation with invalid investor ID"""
    print("\n" + "="*70)
    print("TEST 9: Consult - Invalid Investor")
    print("="*70)
    
    payload = {
        "investor_id": "INVALID999",
        "query": "I want investments"
    }
    
    response = requests.post(f"{BASE_URL}/consult", json=payload)
    assert response.status_code == 404
    
    print(f"✅ Correctly returned 404 for invalid investor")

def test_filter_products_by_risk():
    """Test filtering products by risk"""
    print("\n" + "="*70)
    print("TEST 10: Filter Products by Risk")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/products?risk_level=2")
    assert response.status_code == 200
    
    data = response.json()
    print(f"✅ Low-risk products found: {data['total_count']}")
    
    # Verify all products have risk <= 2
    for product in data['products']:
        assert product['risk_score'] <= 2
    
    print(f"✅ All products have risk ≤ 2")

def test_filter_investors_by_age():
    """Test filtering investors by age"""
    print("\n" + "="*70)
    print("TEST 11: Filter Investors by Age")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/investors?min_age=65")
    assert response.status_code == 200
    
    data = response.json()
    print(f"✅ Elderly investors (65+): {data['total_count']}")
    
    # Verify all investors are 65+
    for investor in data['investors']:
        assert investor['age'] >= 65
    
    print(f"✅ All investors are 65+")

def test_audit_stats():
    """Test audit statistics"""
    print("\n" + "="*70)
    print("TEST 12: Audit Statistics")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/audit/stats")
    assert response.status_code == 200
    
    data = response.json()
    print(f"✅ Total requests: {data['total_requests']}")
    print(f"✅ Approved: {data['approved_count']}")
    print(f"✅ Rejected: {data['rejected_count']}")
    
    if data['total_requests'] > 0:
        print(f"✅ Approval rate: {data['approval_rate']:.1%}")

def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*70)
    print("🧪 RUNNING PHASE 3 API TESTS")
    print("="*70)
    
    tests = [
        test_root,
        test_health,
        test_list_investors,
        test_get_investor,
        test_list_products,
        test_get_product,
        test_consult_young_aggressive,
        test_consult_elderly_conservative,
        test_consult_invalid_investor,
        test_filter_products_by_risk,
        test_filter_investors_by_age,
        test_audit_stats
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
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    print("="*70)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Phase 3 API is working correctly!")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED. Check logs above.")
        return False

if __name__ == "__main__":
    """Run API tests"""
    
    print("\n" + "="*70)
    print("Starting API tests...")
    print("Make sure API is running: docker-compose up")
    print("="*70)
    
    try:
        # Quick health check
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not responding. Start it with: docker-compose up")
            exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Start it with: docker-compose up")
        exit(1)
    
    # Run all tests
    success = run_all_tests()
    
    if success:
        print("\n✅ PHASE 3 API TESTS: SUCCESS\n")
        exit(0)
    else:
        print("\n❌ PHASE 3 API TESTS: FAILED\n")
        exit(1)
