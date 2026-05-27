"""
Database Layer - PostgreSQL Integration
Provides functions to save and retrieve data from PostgreSQL
Handles database-unavailable state gracefully.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from typing import List, Optional, Dict
import os
import json

from app.schemas.schemas import InvestorProfile, Product
from app.core.config import DATABASE_URL

class DatabaseManager:
    """Manages PostgreSQL database connections and operations"""
    
    def __init__(self, database_url: str = None):
        """
        Initialize database connection pool
        """
        if database_url is None:
            database_url = DATABASE_URL
        
        self.database_url = database_url
        self.pool = None
        
        # Create connection pool (reuse connections for performance)
        try:
            self.pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=database_url
            )
            print("✅ Database connection pool created")
        except Exception as e:
            print(f"❌ Failed to create database pool: {e}")
            self.pool = None
    
    def get_connection(self):
        """Get a connection from the pool"""
        if self.pool:
            try:
                return self.pool.getconn()
            except Exception as e:
                print(f"❌ Failed to get connection from pool: {e}")
        return None
    
    def return_connection(self, conn):
        """Return a connection to the pool"""
        if self.pool and conn:
            try:
                self.pool.putconn(conn)
            except Exception as e:
                print(f"❌ Failed to return connection to pool: {e}")
    
    def close_all(self):
        """Close all connections in the pool"""
        if self.pool:
            try:
                self.pool.closeall()
                print("✅ All database connections closed")
            except Exception as e:
                print(f"❌ Failed to close connection pool: {e}")
            self.pool = None
    
    # =========================================================================
    # INVESTOR PROFILE OPERATIONS
    # =========================================================================
    
    def save_investor_profile(self, profile: InvestorProfile) -> bool:
        """
        Save or update an investor profile in PostgreSQL
        """
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Use UPSERT (INSERT ... ON CONFLICT UPDATE)
            sql = """
                INSERT INTO investor_profiles 
                    (investor_id, age, risk_tolerance, annual_income, net_worth, 
                     investment_experience, time_horizon_years)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (investor_id) 
                DO UPDATE SET
                    age = EXCLUDED.age,
                    risk_tolerance = EXCLUDED.risk_tolerance,
                    annual_income = EXCLUDED.annual_income,
                    net_worth = EXCLUDED.net_worth,
                    investment_experience = EXCLUDED.investment_experience,
                    time_horizon_years = EXCLUDED.time_horizon_years,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            cursor.execute(sql, (
                profile.investor_id,
                profile.age,
                profile.risk_tolerance,
                profile.annual_income,
                profile.net_worth,
                profile.investment_experience,
                profile.time_horizon_years
            ))
            
            conn.commit()
            cursor.close()
            
            print(f"✅ Saved investor {profile.investor_id} to database")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save investor {profile.investor_id}: {e}")
            conn.rollback()
            return False
            
        finally:
            self.return_connection(conn)
    
    def get_investor_profile(self, investor_id: str) -> Optional[InvestorProfile]:
        """
        Retrieve an investor profile from PostgreSQL
        """
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            sql = """
                SELECT investor_id, age, risk_tolerance, annual_income, 
                       net_worth, investment_experience, time_horizon_years
                FROM investor_profiles
                WHERE investor_id = %s
            """
            
            cursor.execute(sql, (investor_id,))
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return InvestorProfile(**dict(row))
            return None
            
        except Exception as e:
            print(f"❌ Failed to get investor {investor_id}: {e}")
            return None
            
        finally:
            self.return_connection(conn)
    
    def get_all_investors(self, limit: int = 100) -> List[InvestorProfile]:
        """
        Retrieve all investor profiles from PostgreSQL
        """
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            sql = """
                SELECT investor_id, age, risk_tolerance, annual_income, 
                       net_worth, investment_experience, time_horizon_years
                FROM investor_profiles
                ORDER BY investor_id
                LIMIT %s
            """
            
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
            cursor.close()
            
            return [InvestorProfile(**dict(row)) for row in rows]
            
        except Exception as e:
            print(f"❌ Failed to get all investors: {e}")
            return []
            
        finally:
            self.return_connection(conn)
    
    def search_investors(
        self,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        risk_level: Optional[int] = None,
        experience: Optional[str] = None,
        limit: int = 100
    ) -> List[InvestorProfile]:
        """
        Search investors with filters
        """
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            sql = """
                SELECT investor_id, age, risk_tolerance, annual_income, 
                       net_worth, investment_experience, time_horizon_years
                FROM investor_profiles
                WHERE 1=1
            """
            params = []
            
            if min_age is not None:
                sql += " AND age >= %s"
                params.append(min_age)
            
            if max_age is not None:
                sql += " AND age <= %s"
                params.append(max_age)
            
            if risk_level is not None:
                sql += " AND risk_tolerance = %s"
                params.append(risk_level)
            
            if experience is not None:
                sql += " AND investment_experience = %s"
                params.append(experience)
            
            sql += " ORDER BY investor_id LIMIT %s"
            params.append(limit)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            cursor.close()
            
            return [InvestorProfile(**dict(row)) for row in rows]
            
        except Exception as e:
            print(f"❌ Failed to search investors: {e}")
            return []
            
        finally:
            self.return_connection(conn)
    
    # =========================================================================
    # PRODUCT OPERATIONS
    # =========================================================================
    
    def save_product(self, product: Product) -> bool:
        """
        Save or update a product in PostgreSQL
        """
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            sql = """
                INSERT INTO products 
                    (ticker, name, risk_score, asset_class, min_investment)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (ticker) 
                DO UPDATE SET
                    name = EXCLUDED.name,
                    risk_score = EXCLUDED.risk_score,
                    asset_class = EXCLUDED.asset_class,
                    min_investment = EXCLUDED.min_investment
            """
            
            cursor.execute(sql, (
                product.ticker,
                product.name,
                product.risk_score,
                product.asset_class,
                product.min_investment
            ))
            
            conn.commit()
            cursor.close()
            
            print(f"✅ Saved product {product.ticker} to database")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save product {product.ticker}: {e}")
            conn.rollback()
            return False
            
        finally:
            self.return_connection(conn)
    
    def get_product(self, ticker: str) -> Optional[Product]:
        """
        Retrieve a product from PostgreSQL
        """
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            sql = """
                SELECT ticker, name, risk_score, asset_class, min_investment
                FROM products
                WHERE ticker = %s AND is_active = TRUE
            """
            
            cursor.execute(sql, (ticker,))
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return Product(**dict(row))
            return None
            
        except Exception as e:
            print(f"❌ Failed to get product {ticker}: {e}")
            return None
            
        finally:
            self.return_connection(conn)
    
    def get_all_products(self) -> List[Product]:
        """
        Retrieve all active products from PostgreSQL
        """
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            sql = """
                SELECT ticker, name, risk_score, asset_class, min_investment
                FROM products
                WHERE is_active = TRUE
                ORDER BY risk_score, ticker
            """
            
            cursor.execute(sql)
            rows = cursor.fetchall()
            cursor.close()
            
            return [Product(**dict(row)) for row in rows]
            
        except Exception as e:
            print(f"❌ Failed to get all products: {e}")
            return []
            
        finally:
            self.return_connection(conn)
    
    # =========================================================================
    # BULK OPERATIONS (for migrating from JSON)
    # =========================================================================
    
    def bulk_save_investors(self, profiles: List[InvestorProfile]) -> int:
        """
        Save multiple investor profiles at once
        """
        count = 0
        for profile in profiles:
            if self.save_investor_profile(profile):
                count += 1
        return count
    
    def bulk_save_products(self, products: List[Product]) -> int:
        """
        Save multiple products at once
        """
        count = 0
        for product in products:
            if self.save_product(product):
                count += 1
        return count
    
    # =========================================================================
    # MIGRATION UTILITIES
    # =========================================================================
    
    def migrate_from_json(self, 
                          investors_file: str = 'data/investor_profiles.json',
                          products_file: str = 'data/product_master.json') -> Dict[str, int]:
        """
        Migrate data from JSON files to PostgreSQL
        """
        print("\n" + "="*70)
        print("📦 MIGRATING DATA FROM JSON TO POSTGRESQL")
        print("="*70)
        
        stats = {
            'investors_migrated': 0,
            'products_migrated': 0,
            'errors': []
        }
        
        # Migrate investors
        try:
            if os.path.exists(investors_file):
                with open(investors_file, 'r') as f:
                    investors_data = json.load(f)
                    investors = [InvestorProfile(**p) for p in investors_data]
                    stats['investors_migrated'] = self.bulk_save_investors(investors)
            else:
                stats['errors'].append(f"Investors JSON file {investors_file} not found")
        except Exception as e:
            error = f"Failed to migrate investors: {e}"
            print(f"❌ {error}")
            stats['errors'].append(error)
        
        # Migrate products
        try:
            if os.path.exists(products_file):
                with open(products_file, 'r') as f:
                    products_data = json.load(f)
                    products = [Product(**p) for p in products_data]
                    stats['products_migrated'] = self.bulk_save_products(products)
            else:
                stats['errors'].append(f"Products JSON file {products_file} not found")
        except Exception as e:
            error = f"Failed to migrate products: {e}"
            print(f"❌ {error}")
            stats['errors'].append(error)
        
        return stats

# Global database manager instance
_db_manager = None

def get_db() -> DatabaseManager:
    """Get or create the global database manager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def close_db():
    """Close the global database manager"""
    global _db_manager
    if _db_manager:
        _db_manager.close_all()
        _db_manager = None
