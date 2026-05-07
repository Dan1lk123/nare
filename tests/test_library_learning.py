#!/usr/bin/env python3
"""Test #4: Library Learning - Does rule discovery work?

Hypothesis: System can discover generalizing rules through search.

Test:
- Train on 3 examples of addition
- Discover rule through candidate sampling
- Test on holdout variations
- Check if rule generalizes

Metric:
- Holdout accuracy (rule works on unseen variations)
- Perturbation robustness
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nare.core.evolution.learning import discover_rule


def create_addition_dataset():
    """Create dataset for addition rule discovery."""

    # Training examples
    train = [
        {'query': 'Calculate the sum of 5 and 3', 'solution': '8'},
        {'query': 'What is 2 plus 2?', 'solution': '4'},
        {'query': 'Add 100 and 200', 'solution': '300'},
        {'query': 'Sum of 10 and 15', 'solution': '25'},
        {'query': '7 + 8 equals?', 'solution': '15'},
    ]

    # Holdout examples (unseen variations)
    holdout = [
        {'query': 'Calculate the sum of 12 and 18', 'solution': '30'},
        {'query': 'What is 50 plus 25?', 'solution': '75'},
        {'query': 'Add 3 and 9', 'solution': '12'},
    ]

    return train, holdout


def simple_oracle(query, answer):
    """Simple oracle for testing."""
    # Just check if answer is a number
    try:
        int(answer)
        return True, "Valid number"
    except:
        return False, "Not a number"


def main():
    print("=" * 60)
    print("TEST #4: Library Learning - Rule Discovery")
    print("=" * 60)

    train, holdout = create_addition_dataset()

    print(f"\nTraining examples: {len(train)}")
    for ex in train[:3]:
        print(f"  {ex['query']} -> {ex['solution']}")

    print(f"\nHoldout examples: {len(holdout)}")
    for ex in holdout:
        print(f"  {ex['query']} -> {ex['solution']}")

    # Discover rule
    print("\n" + "=" * 60)
    print("Discovering rule through search...")
    print("=" * 60)

    rule = discover_rule(
        episodes=train + holdout,  # Pass all, function will split
        oracle=simple_oracle,
        n_candidates=5,
        holdout_ratio=0.3
    )

    if rule:
        print("\n[SUCCESS] Rule discovered!")
        print(f"  Pattern: {rule['pattern']}")
        print(f"  Confidence: {rule['confidence']:.2f}")
        print(f"  Holdout score: {rule['holdout_score']:.2f}")
        print(f"\nCode:")
        print(rule['python_code'])

        # Test on new examples
        print("\n" + "=" * 60)
        print("Testing on completely new examples...")
        print("=" * 60)

        test_cases = [
            ('Sum of 99 and 1', '100'),
            ('What is 45 plus 55?', '100'),
            ('Calculate 123 + 456', '579'),
        ]

        trigger_fn = rule['trigger_fn']
        execute_fn = rule['execute_fn']

        correct = 0
        for query, expected in test_cases:
            try:
                if trigger_fn(query):
                    result = execute_fn(query)
                    match = (str(result).strip() == expected)
                    status = "[OK]" if match else "[FAIL]"
                    print(f"{status} {query} -> {result} (expected: {expected})")
                    if match:
                        correct += 1
                else:
                    print(f"[SKIP] {query} (rule didn't trigger)")
            except Exception as e:
                print(f"[ERROR] {query}: {e}")

        print(f"\nTest accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.1%}")

        # Conclusion
        print("\n" + "=" * 60)
        print("CONCLUSION")
        print("=" * 60)

        if correct >= 2:
            print("[OK] Rule GENERALIZES - works on unseen examples")
        else:
            print("[FAIL] Rule does NOT generalize")

    else:
        print("\n[FAIL] No rule discovered")
        print("Possible reasons:")
        print("  - LLM couldn't generate valid candidates")
        print("  - No candidate passed holdout test")
        print("  - Compilation errors")


if __name__ == "__main__":
    main()
