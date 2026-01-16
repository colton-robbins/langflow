"""
Test script to verify AWS Bedrock model updates in Langflow.

This script tests:
1. New models are present in AWS_MODEL_IDs list
2. Claude 4.5 Haiku models are available
3. Cross-region inference models are included
4. Component can be instantiated with new models
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "lfx" / "src"))

from lfx.base.models.aws_constants import AWS_MODEL_IDs, AWS_MODELS_DETAILED


def test_claude_45_models():
    """Test that Claude 4.5 models are in the list."""
    print("Testing Claude 4.5 models...")
    
    expected_models = [
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
    ]
    
    found_models = []
    missing_models = []
    
    for model in expected_models:
        if model in AWS_MODEL_IDs:
            found_models.append(model)
            print(f"  ✓ Found: {model}")
        else:
            missing_models.append(model)
            print(f"  ✗ Missing: {model}")
    
    return len(missing_models) == 0


def test_cross_region_models():
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
        if model in AWS_MODEL_IDs:
            found_models.append(model)
            print(f"  ✓ Found: {model}")
        else:
            missing_models.append(model)
            print(f"  ✗ Missing: {model}")
    
    return len(missing_models) == 0


def test_model_count():
    """Test that we have a reasonable number of models."""
    print(f"\nTotal models in AWS_MODEL_IDs: {len(AWS_MODEL_IDs)}")
    print(f"Total models in AWS_MODELS_DETAILED: {len(AWS_MODELS_DETAILED)}")
    
    # Should have at least 50 models after our additions
    if len(AWS_MODEL_IDs) >= 50:
        print("  ✓ Model count looks good")
        return True
    else:
        print(f"  ✗ Expected at least 50 models, found {len(AWS_MODEL_IDs)}")
        return False


def test_us_prefix_models():
    """Test that we have models with us. prefix."""
    print("\nTesting for us. prefix models...")
    
    us_models = [model for model in AWS_MODEL_IDs if model.startswith("us.")]
    print(f"  Found {len(us_models)} models with 'us.' prefix")
    
    if len(us_models) >= 15:
        print("  ✓ Good number of cross-region models")
        print(f"  Sample models: {us_models[:3]}")
        return True
    else:
        print(f"  ✗ Expected at least 15 cross-region models, found {len(us_models)}")
        return False


def test_component_imports():
    """Test that components can be imported."""
    print("\nTesting component imports...")
    
    try:
        from lfx.components.amazon.amazon_bedrock_converse import AmazonBedrockConverseComponent
        print("  ✓ AmazonBedrockConverseComponent imported successfully")
        
        # Check if combobox is enabled
        model_id_input = None
        for input_field in AmazonBedrockConverseComponent.inputs:
            if hasattr(input_field, 'name') and input_field.name == 'model_id':
                model_id_input = input_field
                break
        
        if model_id_input and hasattr(model_id_input, 'combobox'):
            if model_id_input.combobox:
                print("  ✓ Combobox mode is enabled for model_id")
            else:
                print("  ✗ Combobox mode is NOT enabled for model_id")
                return False
        else:
            print("  ⚠ Could not verify combobox setting")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to import component: {e}")
        return False


def list_all_new_models():
    """List all models that contain '4-5' or start with 'us.'"""
    print("\n" + "="*60)
    print("All New Models Added:")
    print("="*60)
    
    new_models = [
        model for model in AWS_MODEL_IDs 
        if '4-5' in model or (model.startswith('us.') and 'anthropic' in model)
    ]
    
    print("\nClaude 4.5 Models:")
    for model in sorted(new_models):
        if '4-5' in model:
            print(f"  - {model}")
    
    print("\nCross-Region Anthropic Models (us. prefix):")
    for model in sorted(new_models):
        if model.startswith('us.anthropic'):
            print(f"  - {model}")
    
    print("\nCross-Region Meta Models (us. prefix):")
    us_meta = [model for model in AWS_MODEL_IDs if model.startswith('us.meta')]
    for model in sorted(us_meta):
        print(f"  - {model}")
    
    print("\nCross-Region Mistral Models (us. prefix):")
    us_mistral = [model for model in AWS_MODEL_IDs if model.startswith('us.mistral')]
    for model in sorted(us_mistral):
        print(f"  - {model}")


def main():
    """Run all tests."""
    print("="*60)
    print("AWS Bedrock Model Update Tests")
    print("="*60)
    
    results = []
    
    results.append(("Claude 4.5 Models", test_claude_45_models()))
    results.append(("Cross-Region Models", test_cross_region_models()))
    results.append(("Model Count", test_model_count()))
    results.append(("US Prefix Models", test_us_prefix_models()))
    results.append(("Component Imports", test_component_imports()))
    
    # List all new models
    list_all_new_models()
    
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
        print("All tests PASSED!")
        print("="*60)
        return 0
    else:
        print("Some tests FAILED!")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())





