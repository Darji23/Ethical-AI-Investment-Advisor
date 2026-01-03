"""
Phase 2 Test Script - RAG Pipeline Integration Test
Tests the complete RAG pipeline with Jina AI and Fireworks AI

Run this to verify Phase 2 is working correctly
"""
import json
from schemas import InvestorProfile, Product
from vector_store import VectorStore
from jina_embedder import JinaEmbedder
from fireworks_llm import FireworksLLM
from suitability_validator_expanded import SuitabilityValidator
from rag_advisor import RAGInvestmentAdvisor


def test_phase2():
    """Run complete Phase 2 integration test"""
    
    print("\n" + "="*70)
    print("🧪 PHASE 2 INTEGRATION TEST")
    print("Testing: Jina AI + FAISS + Fireworks AI + Phase 1 Validator")
    print("="*70 + "\n")
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    print("📦 STEP 1: Loading data from Phase 1...")
    
    try:
        # Load investors
        with open('investor_profiles.json', 'r') as f:
            investors_data = json.load(f)
            investors = [InvestorProfile(**inv) for inv in investors_data]
        
        # Load products
        with open('product_master.json', 'r') as f:
            products_data = json.load(f)
            products = [Product(**p) for p in products_data]
        
        print(f"✅ Loaded {len(investors)} investors")
        print(f"✅ Loaded {len(products)} products")
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("   Run Phase 1 first: python mock_data_expanded.py")
        return False
    
    # =========================================================================
    # INITIALIZE AI COMPONENTS
    # =========================================================================
    
    print("\n🤖 STEP 2: Initializing AI components...")
    
    try:
        # Initialize Jina embedder
        embedder = JinaEmbedder()
        print("✅ Jina AI embedder ready")
        
        # Initialize Fireworks LLM
        llm = FireworksLLM()
        print("✅ Fireworks AI LLM ready")
        
        # Initialize validator
        validator = SuitabilityValidator(products)
        print("✅ Phase 1 validator ready")
        
    except Exception as e:
        print(f"❌ Failed to initialize AI: {e}")
        print("   Make sure API keys are set:")
        print("   export JINA_API_KEY='your-key'")
        print("   export FIREWORKS_API_KEY='your-key'")
        return False
    
    # =========================================================================
    # LOAD VECTOR STORE
    # =========================================================================
    
    print("\n📊 STEP 3: Loading vector store...")
    
    try:
        vector_store = VectorStore(embedding_dim=1024)
        vector_store.load('product_vector_store')
        print(f"✅ Vector store loaded ({len(vector_store.documents)} products)")
        
    except FileNotFoundError:
        print("❌ Vector store not found!")
        print("   Build it first: python vector_store.py")
        return False
    
    # =========================================================================
    # CREATE RAG ADVISOR
    # =========================================================================
    
    print("\n🎯 STEP 4: Creating RAG advisor...")
    
    advisor = RAGInvestmentAdvisor(
        vector_store=vector_store,
        embedder=embedder,
        llm=llm,
        validator=validator
    )
    
    print("✅ RAG advisor ready")
    
    # =========================================================================
    # TEST SCENARIOS
    # =========================================================================
    
    print("\n" + "="*70)
    print("📋 RUNNING TEST SCENARIOS")
    print("="*70)
    
    test_cases = [
        {
            "name": "Young Aggressive Investor",
            "investor": InvestorProfile(
                investor_id="TEST_YOUNG",
                age=28,
                risk_tolerance=5,
                annual_income=95000,
                net_worth=150000,
                investment_experience="intermediate",
                time_horizon_years=35
            ),
            "query": "I want aggressive growth with tech exposure",
            "expected_status": "APPROVED"
        },
        {
            "name": "Middle-Aged Moderate Investor",
            "investor": InvestorProfile(
                investor_id="TEST_MODERATE",
                age=45,
                risk_tolerance=3,
                annual_income=120000,
                net_worth=400000,
                investment_experience="advanced",
                time_horizon_years=20
            ),
            "query": "I want balanced growth and income",
            "expected_status": "APPROVED"
        },
        {
            "name": "Elderly Conservative Investor",
            "investor": InvestorProfile(
                investor_id="TEST_ELDERLY",
                age=72,
                risk_tolerance=2,
                annual_income=60000,
                net_worth=600000,
                investment_experience="beginner",
                time_horizon_years=5
            ),
            "query": "I need safe stable income for retirement",
            "expected_status": "APPROVED"
        },
        {
            "name": "Beginner Investor (Should avoid complex products)",
            "investor": InvestorProfile(
                investor_id="TEST_BEGINNER",
                age=25,
                risk_tolerance=3,
                annual_income=55000,
                net_worth=20000,
                investment_experience="beginner",
                time_horizon_years=30
            ),
            "query": "I want to start investing",
            "expected_status": "APPROVED"
        }
    ]
    
    # Run tests
    results = []
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"TEST {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'─'*70}")
        
        try:
            result = advisor.get_advice(
                investor_profile=test_case['investor'],
                user_query=test_case['query'],
                top_k=5
            )
            
            # Check result
            status = result['final_status']
            products = result['llm_recommendation'].recommended_products
            allocations = result['llm_recommendation'].allocation_percentages
            rationale = result['llm_recommendation'].rationale
            
            # Verify
            test_passed = True
            issues = []
            
            # Check status
            if status != test_case['expected_status']:
                test_passed = False
                issues.append(f"Expected {test_case['expected_status']}, got {status}")
            
            # Check diversification (if approved)
            if status == "APPROVED":
                if len(products) < 2:
                    test_passed = False
                    issues.append("Less than 2 products (diversification failed)")
                
                # Check allocations sum to 100
                if abs(sum(allocations) - 100) > 0.1:
                    test_passed = False
                    issues.append(f"Allocations sum to {sum(allocations)}%, not 100%")
                
                # Check max allocation
                if max(allocations) > 60:
                    test_passed = False
                    issues.append(f"Max allocation {max(allocations)}% exceeds 60%")
            
            # Print result
            if test_passed:
                print(f"\n✅ TEST PASSED")
                passed += 1
            else:
                print(f"\n❌ TEST FAILED")
                for issue in issues:
                    print(f"   - {issue}")
                failed += 1
            
            print(f"\nResult:")
            print(f"  Status: {status}")
            print(f"  Products: {products}")
            print(f"  Allocations: {allocations}")
            print(f"  Rationale: {rationale[:100]}...")
            
            results.append({
                "test": test_case['name'],
                "passed": test_passed,
                "status": status,
                "products": products,
                "issues": issues
            })
            
        except Exception as e:
            print(f"\n❌ TEST ERROR: {e}")
            failed += 1
            results.append({
                "test": test_case['name'],
                "passed": False,
                "error": str(e)
            })
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(test_cases)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success rate: {passed/len(test_cases)*100:.1f}%")
    print("="*70)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Phase 2 is working correctly!")
        print("\n✅ PHASE 2 INTEGRATION: SUCCESS")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED. Review results above.")
        print("\n❌ PHASE 2 INTEGRATION: NEEDS ATTENTION")
        return False


if __name__ == "__main__":
    """Run Phase 2 integration test"""
    
    success = test_phase2()
    
    if success:
        print("\n" + "="*70)
        print("🎯 PHASE 2 COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("1. Proceed to Phase 3 (Microservices)")
        print("2. Or continue testing with: python rag_advisor.py")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("⚠️  PHASE 2 NEEDS DEBUGGING")
        print("="*70)
        print("\nCheck:")
        print("1. API keys are set (JINA_API_KEY, FIREWORKS_API_KEY)")
        print("2. Vector store exists (python vector_store.py)")
        print("3. Phase 1 data exists (investor_profiles.json, product_master.json)")
        print("="*70 + "\n")
