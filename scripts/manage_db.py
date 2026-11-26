"""
MongoDB Management Utility
Run without external CLI dependencies (uses pymongo from your environment).

Usage:
    python3 -m scripts.manage_db status
    python3 -m scripts.manage_db drop-logs
    python3 -m scripts.manage_db drop-cache
    python3 -m scripts.manage_db seed
    python3 -m scripts.manage_db drop <collection_name>
    python3 -m scripts.manage_db drop-all
"""
import sys
import os
from pymongo import MongoClient

from app.core.config import MONGODB_URL, MONGODB_DB_NAME
from app.core.database import DatabaseManager

def get_client_and_db():
    client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=3000)
    db = client[MONGODB_DB_NAME]
    return client, db

def status():
    try:
        client, db = get_client_and_db()
        client.admin.command('ping')
        print(f"✅ Connected to MongoDB cluster: {MONGODB_URL}")
        print(f"📂 Database: '{MONGODB_DB_NAME}'")
        
        collections = db.list_collection_names()
        if not collections:
            print("   (Database is currently empty)")
            return
            
        print("\n📊 Collections ('Tables') Status:")
        for col in sorted(collections):
            count = db[col].count_documents({})
            print(f"   • {col}: {count} documents")
            
        # Also check legacy 'audit_logs' database if it exists
        if "audit_logs" in client.list_database_names():
            legacy_db = client["audit_logs"]
            legacy_cols = legacy_db.list_collection_names()
            if legacy_cols:
                print("\n📂 Legacy Database: 'audit_logs'")
                for col in sorted(legacy_cols):
                    count = legacy_db[col].count_documents({})
                    print(f"   • {col}: {count} documents")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB at {MONGODB_URL}: {e}")

def drop_collection(name: str):
    try:
        client, db = get_client_and_db()
        if name in db.list_collection_names():
            db[name].drop()
            print(f"🗑️  Dropped collection '{name}' from '{MONGODB_DB_NAME}'")
        else:
            print(f"ℹ️  Collection '{name}' not found in '{MONGODB_DB_NAME}'")
            
        # Also drop from legacy database if it exists
        if "audit_logs" in client.list_database_names() and name in client["audit_logs"].list_collection_names():
            client["audit_logs"][name].drop()
            print(f"🗑️  Dropped legacy collection '{name}' from 'audit_logs' database")
    except Exception as e:
        print(f"❌ Error dropping collection {name}: {e}")

def drop_logs():
    print("🗑️  Dropping audit and violation logs...")
    drop_collection("advisor_logs")
    drop_collection("blocked_recommendations")
    drop_collection("system_events")
    print("✅ Audit logs dropped.")

def drop_cache():
    print("🗑️  Dropping cache collection...")
    drop_collection("cache")
    print("✅ Cache collection dropped.")

def seed():
    print("🌱 Seeding investor profiles and products...")
    dm = DatabaseManager()
    if dm.is_available():
        dm.seed_mock_data()
        print("✅ Seeding complete!")
    else:
        print("❌ MongoDB is currently unavailable. Please verify your connection.")

def drop_all():
    confirm = input(f"Are you sure you want to drop database '{MONGODB_DB_NAME}'? (y/N): ")
    if confirm.strip().lower() == 'y':
        try:
            client, _ = get_client_and_db()
            client.drop_database(MONGODB_DB_NAME)
            print(f"🗑️  Dropped database '{MONGODB_DB_NAME}'")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Cancelled.")

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return

    cmd = sys.argv[1].lower()
    if cmd == "status":
        status()
    elif cmd == "drop-logs":
        drop_logs()
    elif cmd == "drop-cache":
        drop_cache()
    elif cmd == "seed":
        seed()
    elif cmd == "drop-all":
        drop_all()
    elif cmd == "drop" and len(sys.argv) > 2:
        drop_collection(sys.argv[2])
    else:
        print(f"Unknown command: {' '.join(sys.argv[1:])}")
        print(__doc__)

if __name__ == "__main__":
    main()
