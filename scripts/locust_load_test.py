"""
Locust Load Testing Script for Ethical AI Advisor
Simulates realistic user traffic patterns.

Usage:
    locust -f scripts/locust_load_test.py --host=http://localhost:8000
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
    wait_time = between(1, 5)
    
    def on_start(self):
        self.investor_id = random.choice(INVESTOR_IDS)
    
    @task(10)
    @tag('consult', 'main')
    def get_investment_advice(self):
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
    
    @task(3)
    @tag('read', 'investors')
    def view_investor_profile(self):
        with self.client.get(
            f"/investors/{self.investor_id}",
            catch_response=True,
            name="/investors/{id}"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    @tag('read', 'investors')
    def list_investors(self):
        params = {
            "limit": random.choice([5, 10, 20])
        }
        self.client.get("/investors", params=params, name="/investors")
    
    @task(2)
    @tag('read', 'products')
    def view_products(self):
        params = {}
        if random.random() < 0.5:
            params['risk_level'] = random.randint(1, 5)
        self.client.get("/products", params=params, name="/products")
    
    @task(1)
    @tag('audit', 'admin')
    def check_audit_stats(self):
        self.client.get("/audit/stats", name="/audit/stats")
    
    @task(1)
    @tag('health')
    def health_check(self):
        self.client.get("/health", name="/health")


class HeavyLoadUser(HttpUser):
    """
    Simulates aggressive user making many rapid requests
    """
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        self.investor_id = random.choice(INVESTOR_IDS)
    
    @task(1)
    def rapid_fire_requests(self):
        payload = {
            "investor_id": self.investor_id,
            "query": random.choice(QUERIES)
        }
        self.client.post("/consult", json=payload, name="/consult [heavy]")


class ReadOnlyUser(HttpUser):
    """
    Simulates users who only browse, don't request advice
    """
    wait_time = between(2, 8)
    
    @task(5)
    def browse_investors(self):
        self.client.get("/investors?limit=10", name="/investors [readonly]")
    
    @task(5)
    def browse_products(self):
        self.client.get("/products", name="/products [readonly]")
    
    @task(1)
    def check_health(self):
        self.client.get("/health", name="/health [readonly]")
