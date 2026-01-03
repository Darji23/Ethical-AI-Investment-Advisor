"""
Celery configuration for background task processing
Handles audit logging to MongoDB asynchronously
Phase 3 - Integrated with Phase 1 & 2
"""
from celery import Celery
from datetime import datetime
import os
import json

# Initialize Celery
celery_app = Celery(
    'ethical_ai_advisor',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
)

@celery_app.task(name='log_audit_entry')
def log_audit_entry(audit_data: dict):
    """
    Background task to log blocked/approved recommendations to MongoDB
    
    Args:
        audit_data: Dictionary containing request details including:
            - investor_id, investor_age, investor_risk
            - recommendation, validation_result
            - status (APPROVED/REJECTED)
    """
    try:
        from pymongo import MongoClient
        
        # Connect to MongoDB
        mongo_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
        client = MongoClient(mongo_url)
        db = client['audit_logs']
        collection = db['advisor_logs']
        
        # Add server timestamp
        audit_data['server_timestamp'] = datetime.utcnow()
        audit_data['logged_by'] = 'celery_worker'
        
        # Insert into MongoDB
        result = collection.insert_one(audit_data)
        
        print(f"✅ Audit logged to MongoDB: {result.inserted_id}")
        
        # If recommendation was blocked, also log to separate collection
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
        print(f"❌ Failed to log audit entry: {e}")
        # Fallback: Write to local file
        try:
            with open('audit_logs_fallback.jsonl', 'a') as f:
                f.write(json.dumps(audit_data) + '\n')
            print("📝 Logged to fallback file")
        except:
            pass
        raise

@celery_app.task(name='cache_investor_profile')
def cache_investor_profile(investor_id: str):
    """
    Background task to cache frequently accessed investor profiles in Redis
    
    Args:
        investor_id: Investor identifier (INV001-INV100)
    """
    try:
        import redis
        
        # Connect to Redis
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        
        # Load from JSON file (in production would be PostgreSQL)
        with open('investor_profiles.json', 'r') as f:
            profiles = json.load(f)
        
        profile = next((p for p in profiles if p['investor_id'] == investor_id), None)
        
        if profile:
            # Cache in Redis with 1 hour TTL
            cache_key = f"profile:{investor_id}"
            r.setex(cache_key, 3600, json.dumps(profile))
            print(f"✅ Cached profile for {investor_id}")
        else:
            print(f"⚠️  Profile {investor_id} not found")
        
    except Exception as e:
        print(f"❌ Failed to cache profile: {e}")
        raise

@celery_app.task(name='generate_compliance_report')
def generate_compliance_report(start_date: str, end_date: str):
    """
    Background task to generate compliance reports
    
    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
    """
    try:
        from pymongo import MongoClient
        
        mongo_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
        client = MongoClient(mongo_url)
        db = client['audit_logs']
        collection = db['advisor_logs']
        
        # Query logs in date range
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        logs = collection.find({
            'timestamp': {
                '$gte': start_dt.isoformat(),
                '$lte': end_dt.isoformat()
            }
        })
        
        # Generate statistics
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
            
            # Age group stats
            age = log.get('investor_age', 0)
            if age < 35:
                age_groups['young'] += 1
            elif age < 55:
                age_groups['middle'] += 1
            else:
                age_groups['senior'] += 1
            
            # Risk level stats
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
        
        # Save report
        reports_collection = db['compliance_reports']
        result = reports_collection.insert_one(report)
        
        print(f"✅ Compliance report generated: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"❌ Failed to generate report: {e}")
        raise

# Periodic tasks (requires celery beat)
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'daily-compliance-report': {
        'task': 'generate_compliance_report',
        'schedule': crontab(hour=0, minute=0),  # Daily at midnight
        'args': (
            (datetime.utcnow().replace(hour=0, minute=0, second=0).isoformat(),
             datetime.utcnow().isoformat())
        )
    },
}

if __name__ == '__main__':
    celery_app.start()
