"""
Locust Load Testing Script for Ethical AI Advisor
Simulates realistic user traffic patterns

Usage:
    locust -f locust_load_test.py --host=http://localhost:8000
    
    Then open: http://localhost:8089
    Configure: 1000 users, spawn rate 10/sec
"""
from locust import HttpUser, task, between, tag
import random
import json

# Sample investor IDs (INV001-INV100)
INVESTOR_IDS = [f"INV{str(i).zfill(3)}" for i in range(1, 101)]

# Sample investment queries
QUERIES = [
    "I need a balanced portfolio for retirement",
    "I want aggressive growth investments",
    "Give me conservative safe options",
    "I need income-generating investments",
    "What tech stocks should I invest in?",
    "I want to diversify my portfolio",
    "Give me recommendations for long-term growth",
    "I need low-risk stable investments",
    "What are good options for my age and risk level?",
    "I want to maximize returns with moderate risk"
]

class InvestmentAdvisorUser(HttpUser):
    """
    Simulates a user interacting with the investment advisor API
    """
    
    # Wait between 1-5 seconds between tasks (realistic user behavior)
    wait_time = between(1, 5)
    
    def on_start(self):
        """Called when a user starts"""
        # Each user gets a random investor profile
        self.investor_id = random.choice(INVESTOR_IDS)
        print(f"User started with investor_id: {self.investor_id}")
    
    @task(10)  # Weight: 10 (most common task)
    @tag('consult', 'main')
    def get_investment_advice(self):
        """
        Main task: Request investment advice
        This is the core functionality - heaviest load
        """
        payload = {
            "investor_id": self.investor_id,
            "query": random.choice(QUERIES)
        }
        
        with self.client.post(
            "/consult",
            json=payload,
            catch_response=True,
            name="/consult"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get('status') in ['APPROVED', 'REJECTED']:
                    response.success()
                else:
                    response.failure("Invalid status in response")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(3)  # Weight: 3
    @tag('read', 'investors')
    def view_investor_profile(self):
        """
        View own investor profile
        """
        with self.client.get(
            f"/investors/{self.investor_id}",
            catch_response=True,
            name="/investors/{id}"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)  # Weight: 2
    @tag('read', 'investors')
    def list_investors(self):
        """
        Browse investor list (e.g., for comparison)
        """
        params = {
            "limit": random.choice([5, 10, 20])
        }
        self.client.get("/investors", params=params, name="/investors")
    
    @task(2)  # Weight: 2
    @tag('read', 'products')
    def view_products(self):
        """
        Browse available products
        """
        params = {}
        
        # Sometimes filter by risk level
        if random.random() < 0.5:
            params['risk_level'] = random.randint(1, 5)
        
        self.client.get("/products", params=params, name="/products")
    
    @task(1)  # Weight: 1
    @tag('audit', 'admin')
    def check_audit_stats(self):
        """
        Admin checking audit statistics
        Less frequent but important
        """
        self.client.get("/audit/stats", name="/audit/stats")
    
    @task(1)  # Weight: 1
    @tag('health')
    def health_check(self):
        """
        Monitoring systems checking health
        """
        self.client.get("/health", name="/health")
    
    @task(1)  # Weight: 1
    @tag('performance')
    def check_performance_stats(self):
        """
        Check performance metrics
        """
        self.client.get("/performance/stats", name="/performance/stats")


class HeavyLoadUser(HttpUser):
    """
    Simulates aggressive user making many rapid requests
    Use this to stress test the system
    """
    
    wait_time = between(0.1, 0.5)  # Very short wait times
    
    def on_start(self):
        self.investor_id = random.choice(INVESTOR_IDS)
    
    @task(1)
    def rapid_fire_requests(self):
        """Make rapid-fire requests to /consult"""
        payload = {
            "investor_id": self.investor_id,
            "query": random.choice(QUERIES)
        }
        self.client.post("/consult", json=payload, name="/consult [heavy]")


class ReadOnlyUser(HttpUser):
    """
    Simulates users who only browse, don't request advice
    Lighter load on the system
    """
    
    wait_time = between(2, 8)
    
    @task(5)
    def browse_investors(self):
        """Browse investors"""
        self.client.get("/investors?limit=10", name="/investors [readonly]")
    
    @task(5)
    def browse_products(self):
        """Browse products"""
        self.client.get("/products", name="/products [readonly]")
    
    @task(1)
    def check_health(self):
        """Health check"""
        self.client.get("/health", name="/health [readonly]")


# =============================================================================
# Load Test Scenarios
# =============================================================================

"""
SCENARIO 1: Normal Traffic (Default)
-------------------------------------
Run: locust -f locust_load_test.py --host=http://localhost:8000

- Users: 100
- Spawn rate: 10/sec
- Duration: 5 minutes
- Expected: System handles gracefully


SCENARIO 2: High Traffic
-------------------------
Run: locust -f locust_load_test.py --host=http://localhost:8000

- Users: 500
- Spawn rate: 50/sec
- Duration: 10 minutes
- Expected: Some latency increase, but no failures


SCENARIO 3: Stress Test
------------------------
Run: locust -f locust_load_test.py --host=http://localhost:8000 \\
     --users 1000 --spawn-rate 100

- Users: 1000
- Spawn rate: 100/sec
- Duration: Until failure
- Expected: Find breaking point


SCENARIO 4: Spike Test
-----------------------
Run: locust -f locust_load_test.py --host=http://localhost:8000 \\
     --users 2000 --spawn-rate 500

- Users: 2000 (sudden spike)
- Spawn rate: 500/sec (very fast)
- Duration: 2 minutes
- Expected: Test auto-scaling and recovery


SCENARIO 5: Endurance Test
---------------------------
Run: locust -f locust_load_test.py --host=http://localhost:8000 \\
     --users 200 --spawn-rate 20 --run-time 1h

- Users: 200
- Spawn rate: 20/sec
- Duration: 1 hour
- Expected: No memory leaks, stable performance
"""

# =============================================================================
# Expected Results
# =============================================================================

"""
PERFORMANCE TARGETS (Phase 4):

Before Optimization:
- Avg Response Time: ~500ms
- P95: ~1000ms
- P99: ~2000ms
- Max Users: ~100
- Requests/sec: ~200

After Optimization (Phase 4 Target):
- Avg Response Time: <200ms (60% improvement)
- P95: <400ms (60% improvement)
- P99: <800ms (60% improvement)
- Max Users: >1000 (10x improvement)
- Requests/sec: >2000 (10x improvement)

THE DELTA: 60-90% improvement across all metrics! 🚀
"""

# =============================================================================
# Metrics to Watch in Grafana
# =============================================================================

"""
While running Locust, monitor in Grafana:

1. Request Rate: Should match Locust RPS
2. Response Time: Watch P95/P99, should stay < 1s
3. Error Rate: Should stay < 1%
4. Cache Hit Rate: Should be > 90%
5. Active Requests: Monitor queue depth
6. Database Load: Should be low due to caching
"""
