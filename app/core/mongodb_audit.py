"""
MongoDB Audit Logger - Actually Use MongoDB for Audit Trails!
Provides proper audit logging for compliance and regulatory requirements
"""
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, List, Optional, Any
import os

from app.schemas.schemas import InvestorProfile
from app.core.config import MONGODB_URL

class MongoAuditLogger:
    """MongoDB audit logger for compliance tracking"""
    
    def __init__(self, mongo_url: str = None):
        """
        Initialize MongoDB connection
        """
        if mongo_url is None:
            mongo_url = MONGODB_URL
        
        try:
            self.client = MongoClient(mongo_url)
            self.db = self.client['audit_logs']
            
            # Create collections
            self.advisor_logs = self.db['advisor_logs']
            self.blocked_recommendations = self.db['blocked_recommendations']
            self.system_events = self.db['system_events']
            
            # Test connection
            self.client.admin.command('ping')
            self.connected = True
            
            print("✅ MongoDB audit logger connected and ready")
            
            # Create indexes for fast queries
            self._create_indexes()
            
        except Exception as e:
            print(f"⚠️  MongoDB not available: {e}")
            self.client = None
            self.connected = False
    
    def _create_indexes(self):
        """Create indexes for efficient querying"""
        if not self.connected:
            return
        
        try:
            self.advisor_logs.create_index("request_id", unique=True)
            self.advisor_logs.create_index("investor_id")
            self.advisor_logs.create_index("timestamp")
            self.advisor_logs.create_index("status")
            self.advisor_logs.create_index([("investor_age", 1), ("investor_risk", 1)])
            
            self.blocked_recommendations.create_index("request_id")
            self.blocked_recommendations.create_index("timestamp")
            self.blocked_recommendations.create_index("investor_id")
            
            self.system_events.create_index("timestamp")
            self.system_events.create_index("event_type")
            
            print("✅ MongoDB indexes created")
            
        except Exception as e:
            print(f"⚠️  Failed to create indexes: {e}")
    
    def is_available(self) -> bool:
        """Check if MongoDB is available"""
        return self.connected and self.client is not None
    
    # =========================================================================
    # AUDIT LOGGING
    # =========================================================================
    
    def log_recommendation(
        self,
        request_id: str,
        investor_profile: InvestorProfile,
        query: str,
        recommendation: Dict,
        validation_result: Dict,
        status: str,
        performance_metrics: Dict = None
    ) -> bool:
        """
        Log a recommendation to MongoDB
        """
        if not self.is_available():
            return False
        
        try:
            # Build audit log entry
            log_entry = {
                "request_id": request_id,
                "timestamp": datetime.utcnow(),
                "server_timestamp": datetime.utcnow(),
                
                # Investor information
                "investor_id": investor_profile.investor_id,
                "investor_age": investor_profile.age,
                "investor_risk": investor_profile.risk_tolerance,
                "investor_experience": investor_profile.investment_experience,
                "investor_net_worth": investor_profile.net_worth,
                "investor_time_horizon": investor_profile.time_horizon_years,
                
                # Request details
                "query": query,
                "query_length": len(query),
                
                # Recommendation details
                "recommended_products": recommendation.get('recommended_products', []),
                "allocation_percentages": recommendation.get('allocation_percentages', []),
                "rationale": recommendation.get('rationale', ''),
                
                # Validation results
                "status": status,
                "is_suitable": validation_result.get('is_suitable', False),
                "portfolio_risk": validation_result.get('overall_portfolio_risk', 0),
                "approved_products": validation_result.get('approved_products', []),
                "rejected_products": validation_result.get('rejected_products', []),
                "risk_violations": validation_result.get('risk_violations', []),
                "compliance_notes": validation_result.get('compliance_notes', []),
                
                # Performance metrics
                "performance_metrics": performance_metrics or {},
                
                # Metadata
                "logged_by": "api_server",
                "version": "4.1.0"
            }
            
            # Insert into MongoDB
            self.advisor_logs.insert_one(log_entry)
            
            print(f"📝 Logged to MongoDB: {request_id} - {status}")
            
            # If rejected, also log to blocked_recommendations collection
            if status == 'REJECTED':
                self._log_blocked_recommendation(log_entry)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to log recommendation: {e}")
            return False
    
    def _log_blocked_recommendation(self, log_entry: Dict):
        """
        Log blocked recommendations to separate collection for compliance review
        """
        try:
            blocked_entry = {
                "request_id": log_entry['request_id'],
                "timestamp": log_entry['timestamp'],
                "investor_id": log_entry['investor_id'],
                "investor_age": log_entry['investor_age'],
                "investor_risk": log_entry['investor_risk'],
                "query": log_entry['query'],
                "rejected_products": log_entry['rejected_products'],
                "violations": log_entry['risk_violations'],
                "compliance_notes": log_entry['compliance_notes'],
                "severity": self._calculate_severity(log_entry['risk_violations'])
            }
            
            self.blocked_recommendations.insert_one(blocked_entry)
            print(f"⚠️  Blocked recommendation logged: {log_entry['request_id']}")
            
        except Exception as e:
            print(f"❌ Failed to log blocked recommendation: {e}")
    
    def _calculate_severity(self, violations: List[str]) -> str:
        """Calculate severity based on violations"""
        if not violations:
            return "none"
        
        high_severity = ['elderly', 'beginner', 'leveraged', 'crypto', 'exceeded']
        
        for violation in violations:
            if any(keyword in violation.lower() for keyword in high_severity):
                return "high"
        
        if len(violations) >= 3:
            return "medium"
        
        return "low"
    
    # =========================================================================
    # SYSTEM EVENT LOGGING
    # =========================================================================
    
    def log_system_event(
        self,
        event_type: str,
        description: str,
        details: Dict = None,
        severity: str = "info"
    ):
        """
        Log system events (startup, shutdown, errors, etc.)
        """
        if not self.is_available():
            return False
        
        try:
            event = {
                "timestamp": datetime.utcnow(),
                "event_type": event_type,
                "description": description,
                "details": details or {},
                "severity": severity,
                "version": "4.1.0"
            }
            
            self.system_events.insert_one(event)
            print(f"📋 System event logged: {event_type}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to log system event: {e}")
            return False
    
    # =========================================================================
    # QUERY AUDIT LOGS
    # =========================================================================
    
    def query_logs(
        self,
        investor_id: Optional[str] = None,
        status: Optional[str] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        risk_level: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Query audit logs with filters
        """
        if not self.is_available():
            return []
        
        try:
            query_filter = {}
            
            if investor_id:
                query_filter['investor_id'] = investor_id
            
            if status:
                query_filter['status'] = status
            
            if min_age is not None:
                query_filter.setdefault('investor_age', {})['$gte'] = min_age
            
            if max_age is not None:
                query_filter.setdefault('investor_age', {})['$lte'] = max_age
            
            if risk_level is not None:
                query_filter['investor_risk'] = risk_level
            
            if start_date or end_date:
                query_filter.setdefault('timestamp', {})
                if start_date:
                    query_filter['timestamp']['$gte'] = start_date
                if end_date:
                    query_filter['timestamp']['$lte'] = end_date
            
            cursor = self.advisor_logs.find(query_filter).sort('timestamp', -1).limit(limit)
            
            results = []
            for doc in cursor:
                doc.pop('_id', None)
                doc['timestamp'] = doc['timestamp'].isoformat()
                if 'server_timestamp' in doc:
                    doc['server_timestamp'] = doc['server_timestamp'].isoformat()
                results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"❌ Failed to query logs: {e}")
            return []
    
    def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get aggregate statistics from audit logs
        """
        if not self.is_available():
            return {
                "total_requests": 0,
                "approved_count": 0,
                "rejected_count": 0
            }
        
        try:
            date_filter = {}
            if start_date or end_date:
                date_filter['timestamp'] = {}
                if start_date:
                    date_filter['timestamp']['$gte'] = start_date
                if end_date:
                    date_filter['timestamp']['$lte'] = end_date
            
            total = self.advisor_logs.count_documents(date_filter)
            approved = self.advisor_logs.count_documents({**date_filter, 'status': 'APPROVED'})
            rejected = self.advisor_logs.count_documents({**date_filter, 'status': 'REJECTED'})
            
            pipeline = [
                {'$match': date_filter} if date_filter else {'$match': {}},
                {
                    '$group': {
                        '_id': {
                            '$switch': {
                                'branches': [
                                    {'case': {'$lt': ['$investor_age', 35]}, 'then': 'young'},
                                    {'case': {'$lt': ['$investor_age', 55]}, 'then': 'middle'},
                                ],
                                'default': 'senior'
                            }
                        },
                        'count': {'$sum': 1},
                        'approved': {
                            '$sum': {'$cond': [{'$eq': ['$status', 'APPROVED']}, 1, 0]}
                        }
                    }
                }
            ]
            
            age_groups = {}
            for doc in self.advisor_logs.aggregate(pipeline):
                age_group = doc['_id']
                count = doc['count']
                approved_in_group = doc['approved']
                age_groups[age_group] = {
                    'count': count,
                    'approval_rate': approved_in_group / count if count > 0 else 0
                }
            
            pipeline = [
                {'$match': {**date_filter, 'status': 'REJECTED'}},
                {'$unwind': '$risk_violations'},
                {'$group': {'_id': '$risk_violations', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}},
                {'$limit': 5}
            ]
            
            top_violations = [
                {'violation': doc['_id'], 'count': doc['count']}
                for doc in self.advisor_logs.aggregate(pipeline)
            ]
            
            return {
                "total_requests": total,
                "approved_count": approved,
                "rejected_count": rejected,
                "approval_rate": approved / total if total > 0 else 0,
                "rejection_rate": rejected / total if total > 0 else 0,
                "by_age_group": age_groups,
                "top_violations": top_violations
            }
            
        except Exception as e:
            print(f"❌ Failed to get statistics: {e}")
            return {
                "error": str(e),
                "total_requests": 0
            }
    
    def get_blocked_recommendations(self, limit: int = 100) -> List[Dict]:
        """
        Get all blocked recommendations for compliance review
        """
        if not self.is_available():
            return []
        
        try:
            cursor = self.blocked_recommendations.find().sort('timestamp', -1).limit(limit)
            
            results = []
            for doc in cursor:
                doc.pop('_id', None)
                doc['timestamp'] = doc['timestamp'].isoformat()
                results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"❌ Failed to get blocked recommendations: {e}")
            return []
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("✅ MongoDB connection closed")

_mongo_logger = None

def get_mongo_logger() -> MongoAuditLogger:
    """Get or create the global MongoDB logger"""
    global _mongo_logger
    if _mongo_logger is None:
        _mongo_logger = MongoAuditLogger()
    return _mongo_logger

def close_mongo_logger():
    """Close the global MongoDB logger"""
    global _mongo_logger
    if _mongo_logger:
        _mongo_logger.close()
        _mongo_logger = None
