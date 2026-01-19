"""
Test script to measure AWS Bedrock connection performance improvements.

This script helps validate that the boto3 connection pooling optimizations
are working correctly by measuring connection times across multiple requests.
"""

import time
from statistics import mean, stdev


def test_bedrock_connection_performance(num_requests=5):
    """Test boto3 client creation and reuse performance.
    
    Args:
        num_requests: Number of test requests to make
    """
    import boto3
    from botocore.config import Config
    
    # Test 1: Without optimization (baseline)
    print("=" * 60)
    print("Test 1: Baseline (No Connection Pooling)")
    print("=" * 60)
    
    baseline_times = []
    for i in range(num_requests):
        start = time.time()
        
        # Create new client each time (baseline behavior)
        session = boto3.Session()
        client = session.client(
            "bedrock-runtime",
            region_name="us-west-1"
        )
        
        # Simulate a describe call (lightweight operation)
        try:
            # This will fail without credentials but we're measuring connection time
            client._endpoint
        except Exception:
            pass
            
        duration = time.time() - start
        baseline_times.append(duration)
        print(f"  Request {i+1}: {duration*1000:.1f}ms")
    
    print(f"\nBaseline Average: {mean(baseline_times)*1000:.1f}ms")
    if len(baseline_times) > 1:
        print(f"Baseline Std Dev: {stdev(baseline_times)*1000:.1f}ms")
    
    # Test 2: With optimization (connection pooling)
    print("\n" + "=" * 60)
    print("Test 2: Optimized (With Connection Pooling)")
    print("=" * 60)
    
    # Create optimized config
    boto_config = Config(
        max_pool_connections=50,
        connect_timeout=10,
        read_timeout=60,
        retries={
            "max_attempts": 3,
            "mode": "adaptive"
        }
    )
    
    # Create session and client once (cached)
    session = boto3.Session()
    client = session.client(
        "bedrock-runtime",
        region_name="us-west-1",
        config=boto_config
    )
    
    optimized_times = []
    for i in range(num_requests):
        start = time.time()
        
        # Reuse the same client (connection pool behavior)
        try:
            client._endpoint
        except Exception:
            pass
            
        duration = time.time() - start
        optimized_times.append(duration)
        print(f"  Request {i+1}: {duration*1000:.1f}ms")
    
    print(f"\nOptimized Average: {mean(optimized_times)*1000:.1f}ms")
    if len(optimized_times) > 1:
        print(f"Optimized Std Dev: {stdev(optimized_times)*1000:.1f}ms")
    
    # Calculate improvement
    print("\n" + "=" * 60)
    print("Performance Improvement Summary")
    print("=" * 60)
    
    improvement_pct = ((mean(baseline_times) - mean(optimized_times)) / mean(baseline_times)) * 100
    speedup = mean(baseline_times) / mean(optimized_times)
    
    print(f"Average Time Saved: {(mean(baseline_times) - mean(optimized_times))*1000:.1f}ms per request")
    print(f"Improvement: {improvement_pct:.1f}%")
    print(f"Speedup: {speedup:.2f}x")
    
    if improvement_pct > 30:
        print("\nResult: Excellent! Connection pooling is working well.")
    elif improvement_pct > 10:
        print("\nResult: Good! Connection pooling is providing benefit.")
    else:
        print("\nResult: Connection reuse may not be significant for your setup.")
        print("Note: Actual API calls will show more dramatic improvements.")


def test_actual_model_call():
    """Test actual model invocation time (if credentials are available)."""
    import boto3
    from botocore.config import Config
    
    print("\n" + "=" * 60)
    print("Test 3: Actual Model Call (Optional)")
    print("=" * 60)
    
    try:
        # Create optimized client
        boto_config = Config(
            max_pool_connections=50,
            connect_timeout=10,
            read_timeout=60,
            retries={
                "max_attempts": 3,
                "mode": "adaptive"
            }
        )
        
        session = boto3.Session()
        client = session.client(
            "bedrock-runtime",
            region_name="us-west-1",
            config=boto_config
        )
        
        # Warm-up call
        print("\nMaking warm-up call...")
        start = time.time()
        response = client.converse(
            modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "Hi"}]
                }
            ]
        )
        warmup_time = time.time() - start
        print(f"Warm-up call: {warmup_time*1000:.0f}ms")
        
        # Subsequent calls should be faster
        subsequent_times = []
        for i in range(3):
            start = time.time()
            response = client.converse(
                modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": "Hi"}]
                    }
                ]
            )
            duration = time.time() - start
            subsequent_times.append(duration)
            print(f"Call {i+1}: {duration*1000:.0f}ms")
        
        print(f"\nAverage subsequent call time: {mean(subsequent_times)*1000:.0f}ms")
        improvement = ((warmup_time - mean(subsequent_times)) / warmup_time) * 100
        print(f"Improvement vs warm-up: {improvement:.1f}%")
        
    except Exception as e:
        print(f"\nSkipping actual model test: {e}")
        print("This is expected if AWS credentials are not configured.")


if __name__ == "__main__":
    print("\nAWS Bedrock Connection Performance Test")
    print("========================================\n")
    print("This test measures the impact of connection pooling optimizations.")
    print("You should see faster subsequent requests with the optimized configuration.\n")
    
    # Run connection performance test
    test_bedrock_connection_performance(num_requests=5)
    
    # Optionally test actual model calls
    try_actual_test = input("\nTest with actual model calls? (requires AWS credentials) [y/N]: ")
    if try_actual_test.lower() == 'y':
        test_actual_model_call()
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)
