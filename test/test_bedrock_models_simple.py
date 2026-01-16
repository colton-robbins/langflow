"""
Simple test script to verify AWS Bedrock model updates in Langflow.
This version directly reads the constants file without importing dependencies.
"""

import re
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def read_aws_constants():
    """Read and parse the aws_constants.py file."""
    constants_file = Path(__file__).parent.parent / "src" / "lfx" / "src" / "lfx" / "base" / "models" / "aws_constants.py"
    
    if not constants_file.exists():
        print(f"Error: Could not find {constants_file}")
        return []
    
    with open(constants_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract model names using regex
    pattern = r'name="([^"]+)"'
    models = re.findall(pattern, content)
    
    return models


def test_claude_45_models(models):
    """Test that Claude 4.5 models are in the list."""
    print("Testing Claude 4.5 models...")
    
    expected_models = [
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
    ]
    
    found_models = []
    missing_models = []
    
    for model in expected_models:
        if model in models:
            found_models.append(model)
            print(f"  ✓ Found: {model}")
        else:
            missing_models.append(model)
            print(f"  ✗ Missing: {model}")
    
    return len(missing_models) == 0


def test_cross_region_models(models):
    """Test that cross-region inference models are in the list."""
    print("\nTesting cross-region inference models...")
    
    # Test a sample of cross-region models
    expected_models = [
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "us.meta.llama3-1-405b-instruct-v1:0",
        "us.mistral.mistral-large-2407-v1:0",
    ]
    
    found_models = []
    missing_models = []
    
    for model in expected_models:
        if model in models:
            found_models.append(model)
            print(f"  ✓ Found: {model}")
        else:
            missing_models.append(model)
            print(f"  ✗ Missing: {model}")
    
    return len(missing_models) == 0


def test_model_count(models):
    """Test that we have a reasonable number of models."""
    print(f"\nTotal models found: {len(models)}")
    
    # Should have at least 50 models after our additions
    if len(models) >= 50:
        print("  ✓ Model count looks good")
        return True
    else:
        print(f"  ✗ Expected at least 50 models, found {len(models)}")
        return False


def test_us_prefix_models(models):
    """Test that we have models with us. prefix."""
    print("\nTesting for us. prefix models...")
    
    us_models = [model for model in models if model.startswith("us.")]
    print(f"  Found {len(us_models)} models with 'us.' prefix")
    
    if len(us_models) >= 15:
        print("  ✓ Good number of cross-region models")
        print(f"  Sample models: {us_models[:3]}")
        return True
    else:
        print(f"  ✗ Expected at least 15 cross-region models, found {len(us_models)}")
        return False


def test_combobox_enabled():
    """Test that combobox is enabled in the component files."""
    print("\nTesting combobox configuration...")
    
    files_to_check = [
        Path(__file__).parent.parent / "src" / "lfx" / "src" / "lfx" / "components" / "amazon" / "amazon_bedrock_converse.py",
        Path(__file__).parent.parent / "src" / "lfx" / "src" / "lfx" / "components" / "amazon" / "amazon_bedrock_model.py",
    ]
    
    all_good = True
    for file_path in files_to_check:
        if not file_path.exists():
            print(f"  ✗ File not found: {file_path.name}")
            all_good = False
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if combobox=True is present in model_id dropdown
        if 'name="model_id"' in content and 'combobox=True' in content:
            print(f"  ✓ Combobox enabled in {file_path.name}")
        else:
            print(f"  ✗ Combobox NOT enabled in {file_path.name}")
            all_good = False
    
    return all_good


def list_all_new_models(models):
    """List all models that contain '4-5' or start with 'us.'"""
    print("\n" + "="*60)
    print("All New Models Added:")
    print("="*60)
    
    print("\nClaude 4.5 Models:")
    claude_45 = [model for model in models if '4-5' in model]
    for model in sorted(claude_45):
        print(f"  - {model}")
    
    print("\nCross-Region Anthropic Models (us. prefix):")
    us_anthropic = [model for model in models if model.startswith('us.anthropic')]
    for model in sorted(us_anthropic):
        print(f"  - {model}")
    
    print("\nCross-Region Meta Models (us. prefix):")
    us_meta = [model for model in models if model.startswith('us.meta')]
    for model in sorted(us_meta):
        print(f"  - {model}")
    
    print("\nCross-Region Mistral Models (us. prefix):")
    us_mistral = [model for model in models if model.startswith('us.mistral')]
    for model in sorted(us_mistral):
        print(f"  - {model}")
    
    print(f"\nTotal new models with 'us.' prefix: {len(us_anthropic) + len(us_meta) + len(us_mistral)}")
    print(f"Total Claude 4.5 models: {len(claude_45)}")


def main():
    """Run all tests."""
    print("="*60)
    print("AWS Bedrock Model Update Tests (Simple)")
    print("="*60)
    
    # Read models from file
    models = read_aws_constants()
    
    if not models:
        print("Error: Could not read models from aws_constants.py")
        return 1
    
    results = []
    
    results.append(("Claude 4.5 Models", test_claude_45_models(models)))
    results.append(("Cross-Region Models", test_cross_region_models(models)))
    results.append(("Model Count", test_model_count(models)))
    results.append(("US Prefix Models", test_us_prefix_models(models)))
    results.append(("Combobox Configuration", test_combobox_enabled()))
    
    # List all new models
    list_all_new_models(models)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests PASSED!")
        print("="*60)
        print("\nYou can now use the new Bedrock models in Langflow!")
        print("Remember to:")
        print("  1. Restart Langflow to apply changes")
        print("  2. Enable model access in AWS Bedrock console")
        print("  3. Configure AWS credentials")
        return 0
    else:
        print("✗ Some tests FAILED!")
        print("="*60)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

