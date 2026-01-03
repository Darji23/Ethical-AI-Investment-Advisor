"""
Performance Benchmark Script - Measure "The Delta"
Compares Phase 3 (no optimization) vs Phase 4 (optimized)

This script proves the performance improvements Microsoft wants to see!
"""
import requests
import time
import statistics
import json
from datetime import datetime
from typing import List, Dict
import sys

BASE_URL = "http://localhost:8000"

# Sample test data
TEST_INVESTORS = [f"INV{str(i).zfill(3)}" for i in range(1, 21)]  # First 20 investors
TEST_QUERIES = [
    "I need a balanced portfolio for retirement",
    "I want aggressive growth investments",
    "Give me conservative safe options",
]

class PerformanceBenchmark:
    """Benchmark performance metrics"""
    
    def __init__(self):
        self.results = {
            'response_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'successful_requests': 0
        }
    
    def test_single_request(self, investor_id: str, query: str) -> Dict:
        """Test a single request and measure performance"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{BASE_URL}/consult",
                json={
                    "investor_id": investor_id,
                    "query": query
                },
                timeout=10
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.results['successful_requests'] += 1
                
                # Check if cached
                if 'performance_metrics' in data:
                    if data['performance_metrics'].get('cached', False):
                        self.results['cache_hits'] += 1
                    else:
                        self.results['cache_misses'] += 1
                
                return {
                    'success': True,
                    'duration': duration,
                    'status': data.get('status'),
                    'cached': data.get('performance_metrics', {}).get('cached', False)
                }
            else:
                self.results['errors'] += 1
                return {
                    'success': False,
                    'duration': duration,
                    'error': f"Status {response.status_code}"
                }
                
        except Exception as e:
            self.results['errors'] += 1
            return {
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    def run_benchmark(self, num_requests: int = 100, warmup: int = 10):
        """Run benchmark with specified number of requests"""
        print(f"\n{'='*70}")
        print(f"STARTING BENCHMARK: {num_requests} requests")
        print(f"{'='*70}\n")
        
        # Warmup phase (to populate cache)
        print(f"🔥 Warming up cache ({warmup} requests)...")
        for i in range(warmup):
            investor_id = TEST_INVESTORS[i % len(TEST_INVESTORS)]
            query = TEST_QUERIES[i % len(TEST_QUERIES)]
            self.test_single_request(investor_id, query)
        
        # Reset results after warmup
        self.results = {
            'response_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'successful_requests': 0
        }
        
        # Actual benchmark
        print(f"📊 Running benchmark ({num_requests} requests)...\n")
        
        start_benchmark = time.time()
        
        for i in range(num_requests):
            # Use same investors repeatedly to test caching
            investor_id = TEST_INVESTORS[i % len(TEST_INVESTORS)]
            query = TEST_QUERIES[i % len(TEST_QUERIES)]
            
            result = self.test_single_request(investor_id, query)
            
            if result['success']:
                self.results['response_times'].append(result['duration'])
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_requests} requests completed", end='\r')
        
        total_benchmark_time = time.time() - start_benchmark
        print(f"\n\n✅ Benchmark completed in {total_benchmark_time:.2f} seconds")
        
        return self.calculate_statistics()
    
    def calculate_statistics(self) -> Dict:
        """Calculate performance statistics"""
        if not self.results['response_times']:
            return None
        
        response_times = self.results['response_times']
        
        stats = {
            'total_requests': len(response_times) + self.results['errors'],
            'successful_requests': self.results['successful_requests'],
            'failed_requests': self.results['errors'],
            'avg_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'p95_response_time': self.percentile(response_times, 95),
            'p99_response_time': self.percentile(response_times, 99),
            'stddev_response_time': statistics.stdev(response_times) if len(response_times) > 1 else 0,
            'cache_hits': self.results['cache_hits'],
            'cache_misses': self.results['cache_misses'],
            'cache_hit_rate': self.results['cache_hits'] / (self.results['cache_hits'] + self.results['cache_misses']) if (self.results['cache_hits'] + self.results['cache_misses']) > 0 else 0,
            'requests_per_second': self.results['successful_requests'] / sum(response_times) if response_times else 0
        }
        
        return stats
    
    @staticmethod
    def percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def print_results(self, stats: Dict):
        """Print benchmark results in a nice format"""
        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS")
        print(f"{'='*70}\n")
        
        print(f"📊 REQUEST STATISTICS")
        print(f"{'─'*70}")
        print(f"  Total Requests:      {stats['total_requests']}")
        print(f"  Successful:          {stats['successful_requests']} ({stats['successful_requests']/stats['total_requests']*100:.1f}%)")
        print(f"  Failed:              {stats['failed_requests']} ({stats['failed_requests']/stats['total_requests']*100:.1f}%)")
        print()
        
        print(f"⚡ PERFORMANCE METRICS")
        print(f"{'─'*70}")
        print(f"  Avg Response Time:   {stats['avg_response_time']*1000:.2f} ms")
        print(f"  Median Response:     {stats['median_response_time']*1000:.2f} ms")
        print(f"  Min Response:        {stats['min_response_time']*1000:.2f} ms")
        print(f"  Max Response:        {stats['max_response_time']*1000:.2f} ms")
        print(f"  P95 Response:        {stats['p95_response_time']*1000:.2f} ms")
        print(f"  P99 Response:        {stats['p99_response_time']*1000:.2f} ms")
        print(f"  Std Deviation:       {stats['stddev_response_time']*1000:.2f} ms")
        print()
        
        print(f"💾 CACHE PERFORMANCE")
        print(f"{'─'*70}")
        print(f"  Cache Hits:          {stats['cache_hits']}")
        print(f"  Cache Misses:        {stats['cache_misses']}")
        print(f"  Cache Hit Rate:      {stats['cache_hit_rate']*100:.1f}%")
        print()
        
        print(f"🚀 THROUGHPUT")
        print(f"{'─'*70}")
        print(f"  Requests/Second:     {stats['requests_per_second']:.2f}")
        print()
        
        # Performance grade
        avg_ms = stats['avg_response_time'] * 1000
        if avg_ms < 100:
            grade = "⭐⭐⭐⭐⭐ EXCELLENT"
        elif avg_ms < 200:
            grade = "⭐⭐⭐⭐ GREAT"
        elif avg_ms < 500:
            grade = "⭐⭐⭐ GOOD"
        elif avg_ms < 1000:
            grade = "⭐⭐ ACCEPTABLE"
        else:
            grade = "⭐ NEEDS IMPROVEMENT"
        
        print(f"📈 PERFORMANCE GRADE: {grade}")
        print(f"{'='*70}\n")
    
    def compare_with_baseline(self, stats: Dict, baseline: Dict):
        """Compare current performance with baseline (The Delta!)"""
        print(f"\n{'='*70}")
        print("📊 PERFORMANCE COMPARISON - THE DELTA!")
        print(f"{'='*70}\n")
        
        metrics = [
            ('Avg Response Time', 'avg_response_time', 'ms', 1000, True),
            ('P95 Response Time', 'p95_response_time', 'ms', 1000, True),
            ('P99 Response Time', 'p99_response_time', 'ms', 1000, True),
            ('Requests/Second', 'requests_per_second', '', 1, False),
            ('Cache Hit Rate', 'cache_hit_rate', '%', 100, False)
        ]
        
        print(f"{'Metric':<25} {'Before':<15} {'After':<15} {'Delta':<15}")
        print(f"{'─'*70}")
        
        for metric_name, key, unit, multiplier, lower_is_better in metrics:
            before = baseline.get(key, 0) * multiplier
            after = stats.get(key, 0) * multiplier
            
            if before > 0:
                if lower_is_better:
                    improvement = ((before - after) / before) * 100
                    delta_symbol = "↓" if after < before else "↑"
                else:
                    improvement = ((after - before) / before) * 100
                    delta_symbol = "↑" if after > before else "↓"
                
                delta_str = f"{delta_symbol} {abs(improvement):.1f}%"
            else:
                delta_str = "N/A"
            
            before_str = f"{before:.2f} {unit}" if unit else f"{before:.2f}"
            after_str = f"{after:.2f} {unit}" if unit else f"{after:.2f}"
            
            print(f"{metric_name:<25} {before_str:<15} {after_str:<15} {delta_str:<15}")
        
        print(f"{'='*70}\n")
        
        # Overall assessment
        avg_improvement = ((baseline['avg_response_time'] - stats['avg_response_time']) / baseline['avg_response_time']) * 100
        
        if avg_improvement > 50:
            print(f"🚀 OUTSTANDING! {avg_improvement:.0f}% performance improvement!")
        elif avg_improvement > 25:
            print(f"✅ EXCELLENT! {avg_improvement:.0f}% performance improvement!")
        elif avg_improvement > 10:
            print(f"👍 GOOD! {avg_improvement:.0f}% performance improvement!")
        elif avg_improvement > 0:
            print(f"📈 MINOR IMPROVEMENT: {avg_improvement:.0f}%")
        else:
            print(f"⚠️  PERFORMANCE REGRESSION: {abs(avg_improvement):.0f}%")
        
        print()

def run_full_benchmark():
    """Run complete benchmark suite"""
    print("\n" + "="*70)
    print("ETHICAL AI ADVISOR - PERFORMANCE BENCHMARK")
    print("Phase 4: Measuring The Delta")
    print("="*70)
    
    # Check if service is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ ERROR: API service is not healthy!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Cannot connect to API at {BASE_URL}")
        print(f"   Make sure the service is running: docker-compose up")
        sys.exit(1)
    
    print("\n✅ API service is running\n")
    
    # Run benchmark
    benchmark = PerformanceBenchmark()
    stats = benchmark.run_benchmark(num_requests=100, warmup=20)
    
    if stats:
        benchmark.print_results(stats)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"💾 Results saved to: {filename}\n")
        
        # Check if baseline exists for comparison
        try:
            with open('benchmark_baseline.json', 'r') as f:
                baseline = json.load(f)
            benchmark.compare_with_baseline(stats, baseline)
        except FileNotFoundError:
            print("💡 TIP: Save this as baseline for future comparisons:")
            print(f"   cp {filename} benchmark_baseline.json\n")

if __name__ == "__main__":
    run_full_benchmark()
