"""
Celery configuration for background task processing
Handles audit logging to MongoDB asynchronously
"""
from celery import Celery
from datetime import datetime
import os
import json

# Initialize Celery
mongo_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
celery_app = Celery(
    'ethical_ai_advisor',
    broker=os.getenv('CELERY_BROKER_URL', f"{mongo_url}/celery_broker"),
    backend=os.getenv('CELERY_RESULT_BACKEND', f"{mongo_url}/celery_backend")
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
)

@celery_app.task(name='log_audit_entry')
def log_audit_entry(audit_data: dict):
    """
    Background task to log blocked/approved recommendations to MongoDB
    """
    try:
        from pymongo import MongoClient
        from app.core.config import MONGODB_URL, MONGODB_DB_NAME
        
        client = MongoClient(MONGODB_URL)
        db = client[MONGODB_DB_NAME]
        collection = db['advisor_logs']
        
        audit_data['server_timestamp'] = datetime.utcnow()
        audit_data['logged_by'] = 'celery_worker'
        
        result = collection.insert_one(audit_data)
        print(f"✅ Audit logged to MongoDB: {result.inserted_id}")
        
        if audit_data.get('status') == 'REJECTED':
            blocked_collection = db['blocked_recommendations']
            blocked_data = {
                'request_id': audit_data['request_id'],
                'investor_id': audit_data['investor_id'],
                'investor_age': audit_data.get('investor_age'),
                'investor_risk': audit_data.get('investor_risk'),
                'violations': audit_data.get('violations', []),
                'blocked_products': audit_data.get('blocked_products', []),
                'timestamp': audit_data['timestamp'],
                'server_timestamp': audit_data['server_timestamp']
            }
            blocked_collection.insert_one(blocked_data)
            print(f"⚠️  Blocked recommendation logged: {audit_data['request_id']}")
        
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"❌ Failed to log audit entry to MongoDB: {e}")
        raise

@celery_app.task(name='cache_investor_profile')
def cache_investor_profile(investor_id: str):
    """
    Background task to cache frequently accessed investor profiles in MongoDB cache
    """
    try:
        from app.core.cache import get_cache
        from app.core.database import DatabaseManager
        
        db = DatabaseManager()
        if not db.is_available():
            print(f"⚠️  MongoDB not available to cache profile {investor_id}")
            return
            
        profile = db.get_investor_profile(investor_id)
        if profile:
            get_cache().cache_investor(profile, ttl_seconds=3600)
            print(f"✅ Cached profile for {investor_id} in MongoDB cache")
        else:
            print(f"⚠️  Profile {investor_id} not found in MongoDB")
        db.close()
        
    except Exception as e:
        print(f"❌ Failed to cache profile: {e}")
        raise

@celery_app.task(name='generate_compliance_report')
def generate_compliance_report(start_date: str, end_date: str):
    """
    Background task to generate compliance reports
    """
    try:
        from pymongo import MongoClient
        from app.core.config import MONGODB_URL, MONGODB_DB_NAME
        client = MongoClient(MONGODB_URL)
        db = client[MONGODB_DB_NAME]
        collection = db['advisor_logs']
        
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        logs = collection.find({
            'timestamp': {
                '$gte': start_dt.isoformat(),
                '$lte': end_dt.isoformat()
            }
        })
        
        total = 0
        approved = 0
        rejected = 0
        violations = {}
        age_groups = {'young': 0, 'middle': 0, 'senior': 0}
        risk_levels = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for log in logs:
            total += 1
            if log['status'] == 'APPROVED':
                approved += 1
            else:
                rejected += 1
                for violation in log.get('violations', []):
                    violations[violation] = violations.get(violation, 0) + 1
            
            age = log.get('investor_age', 0)
            if age < 35:
                age_groups['young'] += 1
            elif age < 55:
                age_groups['middle'] += 1
            else:
                age_groups['senior'] += 1
            
            risk = log.get('investor_risk', 0)
            if 1 <= risk <= 5:
                risk_levels[risk] += 1
        
        report = {
            'report_generated_at': datetime.utcnow().isoformat(),
            'period': {
                'start': start_date,
                'end': end_date
            },
            'statistics': {
                'total_requests': total,
                'approved': approved,
                'rejected': rejected,
                'approval_rate': approved / total if total > 0 else 0,
                'rejection_rate': rejected / total if total > 0 else 0
            },
            'by_age_group': age_groups,
            'by_risk_level': risk_levels,
            'top_violations': sorted(
                violations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
        
        reports_collection = db['compliance_reports']
        result = reports_collection.insert_one(report)
        print(f"✅ Compliance report generated: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"❌ Failed to generate report: {e}")
        raise

# Periodic tasks
from celery.schedules import crontab
celery_app.conf.beat_schedule = {
    'daily-compliance-report': {
        'task': 'generate_compliance_report',
        'schedule': crontab(hour=0, minute=0),
        'args': (
            (datetime.utcnow().replace(hour=0, minute=0, second=0).isoformat(),
             datetime.utcnow().isoformat())
        )
    },
}

if __name__ == '__main__':
    celery_app.start()
