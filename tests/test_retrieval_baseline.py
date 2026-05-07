#!/usr/bin/env python3
"""Test #1: Retrieval - Is memory useful or just noise?

Hypothesis: Similar tasks help solve new tasks.

Test:
- Baseline: Random answer (no memory)
- Memory-only: kNN retrieval → nearest neighbor answer (no LLM)
- Full system: Normal NARE with LLM

Metric:
- Accuracy on similar tasks
- Degradation on novel tasks
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nare.core.agent import NAREProductionAgent
from nare.config import DEFAULT_CONFIG
from nare import llm


def create_test_dataset():
    """Create a small dataset with:
    - Repeated tasks (exact duplicates)
    - Similar tasks (variations)
    - Novel tasks (completely different)
    """
    tasks = []

    # Group 1: Repeated tasks (exact duplicates)
    for i in range(3):
        tasks.append({
            'id': f'repeat_{i}',
            'query': 'Calculate the sum of 5 and 3',
            'expected': '8',
            'type': 'repeat'
        })

    # Group 2: Similar tasks (variations of addition)
    similar_queries = [
        ('Calculate the sum of 10 and 7', '17'),
        ('What is 15 plus 8?', '23'),
        ('Add 20 and 12', '32'),
    ]
    for i, (query, expected) in enumerate(similar_queries):
        tasks.append({
            'id': f'similar_{i}',
            'query': query,
            'expected': expected,
            'type': 'similar'
        })

    # Group 3: Novel tasks (completely different)
    novel_queries = [
        ('What is the capital of France?', 'Paris'),
        ('Reverse the string "hello"', 'olleh'),
        ('Is 17 a prime number?', 'yes'),
    ]
    for i, (query, expected) in enumerate(novel_queries):
        tasks.append({
            'id': f'novel_{i}',
            'query': query,
            'expected': expected,
            'type': 'novel'
        })

    return tasks


def baseline_random(tasks):
    """Baseline: Random answer (no memory, no reasoning)"""
    results = []

    for task in tasks:
        # Random answer from a pool
        random_answers = ['8', '17', '23', '32', 'Paris', 'olleh', 'yes', 'no', '42']
        answer = random.choice(random_answers)

        correct = (answer == task['expected'])
        results.append({
            'id': task['id'],
            'type': task['type'],
            'correct': correct,
            'answer': answer,
            'expected': task['expected']
        })

    return results


def memory_only_knn(tasks, agent):
    """Memory-only: kNN retrieval → nearest neighbor answer (no LLM)"""
    results = []

    for task in tasks:
        query = task['query']

        # Get embedding
        query_emb = llm.get_embedding(query)
        query_emb_np = np.array([query_emb], dtype=np.float32)

        # Retrieve nearest neighbor
        retrieved = agent.memory.retrieve_episodes(query_emb_np, k=1)

        if retrieved and len(retrieved) > 0:
            # Use answer from nearest neighbor
            answer = retrieved[0].get('answer', 'NO_ANSWER')
            similarity = retrieved[0].get('similarity', 0.0)
        else:
            answer = 'NO_MEMORY'
            similarity = 0.0

        correct = (answer == task['expected'])
        results.append({
            'id': task['id'],
            'type': task['type'],
            'correct': correct,
            'answer': answer,
            'expected': task['expected'],
            'similarity': float(similarity)
        })

    return results


def full_system(tasks, agent):
    """Full system: Normal NARE with LLM"""
    results = []

    for task in tasks:
        query = task['query']

        # Solve with full NARE
        result = agent.solve(query)
        answer = result.get('final_answer', 'NO_ANSWER')
        route = result.get('route_decision', 'UNKNOWN')

        correct = (answer.strip() == task['expected'])
        results.append({
            'id': task['id'],
            'type': task['type'],
            'correct': correct,
            'answer': answer,
            'expected': task['expected'],
            'route': route
        })

    return results


def analyze_results(results, method_name):
    """Analyze results by task type"""
    by_type = {}

    for r in results:
        task_type = r['type']
        if task_type not in by_type:
            by_type[task_type] = {'correct': 0, 'total': 0}

        by_type[task_type]['total'] += 1
        if r['correct']:
            by_type[task_type]['correct'] += 1

    print(f"\n{method_name}:")
    print("=" * 60)

    total_correct = 0
    total_tasks = 0

    for task_type, stats in by_type.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {task_type:10s}: {stats['correct']}/{stats['total']} = {accuracy:.1%}")
        total_correct += stats['correct']
        total_tasks += stats['total']

    overall = total_correct / total_tasks if total_tasks > 0 else 0
    print(f"  {'Overall':10s}: {total_correct}/{total_tasks} = {overall:.1%}")

    return by_type


def seed_memory_with_training_data(agent):
    """Seed memory with some training examples"""
    training_data = [
        ('Calculate the sum of 5 and 3', '8'),
        ('What is 2 plus 2?', '4'),
        ('Add 100 and 200', '300'),
    ]

    print("\nSeeding memory with training data...")
    for query, answer in training_data:
        # Create episode
        query_emb = llm.get_embedding(query)  # Returns list

        episode = {
            'query': query,
            'answer': answer,
            'solution': answer,  # Add 'solution' field for compatibility
            'score': 1.0,
            'epoch': 0,
            'activation_score': 1.0,
            'access_count': 0,
            'last_used_epoch': 0,
        }

        # add_episode(episode_data, embedding) - correct order
        agent.memory.add_episode(episode, query_emb)

    agent.memory.save()
    print(f"Seeded {len(training_data)} examples into memory")


def main():
    print("=" * 60)
    print("TEST #1: Retrieval - Is memory useful or just noise?")
    print("=" * 60)

    # Create test dataset
    tasks = create_test_dataset()
    print(f"\nCreated test dataset: {len(tasks)} tasks")
    print(f"  - Repeat: {len([t for t in tasks if t['type'] == 'repeat'])}")
    print(f"  - Similar: {len([t for t in tasks if t['type'] == 'similar'])}")
    print(f"  - Novel: {len([t for t in tasks if t['type'] == 'novel'])}")

    # Initialize agent (evolution will run in background, ignore errors)
    agent = NAREProductionAgent(
        config=DEFAULT_CONFIG,
        persist_dir="memory_retrieval_test",
        embedding_dim=3072  # Match get_embedding() output
    )

    # Seed memory with training data
    seed_memory_with_training_data(agent)

    # Test 1: Baseline (random)
    print("\n" + "=" * 60)
    print("Running Baseline (random)...")
    baseline_results = baseline_random(tasks)
    baseline_stats = analyze_results(baseline_results, "Baseline (Random)")

    # Test 2: Memory-only (kNN)
    print("\n" + "=" * 60)
    print("Running Memory-only (kNN)...")
    memory_results = memory_only_knn(tasks, agent)
    memory_stats = analyze_results(memory_results, "Memory-only (kNN)")

    # Test 3: Full system (with LLM)
    print("\n" + "=" * 60)
    print("Running Full system (NARE + LLM)...")
    full_results = full_system(tasks, agent)
    full_stats = analyze_results(full_results, "Full System (NARE)")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)

    for task_type in ['repeat', 'similar', 'novel']:
        print(f"\n{task_type.upper()} tasks:")

        baseline_acc = baseline_stats.get(task_type, {}).get('correct', 0) / baseline_stats.get(task_type, {}).get('total', 1)
        memory_acc = memory_stats.get(task_type, {}).get('correct', 0) / memory_stats.get(task_type, {}).get('total', 1)
        full_acc = full_stats.get(task_type, {}).get('correct', 0) / full_stats.get(task_type, {}).get('total', 1)

        print(f"  Baseline:     {baseline_acc:.1%}")
        print(f"  Memory-only:  {memory_acc:.1%} (Delta {memory_acc - baseline_acc:+.1%})")
        print(f"  Full system:  {full_acc:.1%} (Delta {full_acc - baseline_acc:+.1%})")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    memory_gain = memory_stats.get('similar', {}).get('correct', 0) / memory_stats.get('similar', {}).get('total', 1)
    baseline_similar = baseline_stats.get('similar', {}).get('correct', 0) / baseline_stats.get('similar', {}).get('total', 1)

    if memory_gain > baseline_similar + 0.2:
        print("[OK] Memory is USEFUL - significant gain on similar tasks")
    elif memory_gain > baseline_similar:
        print("[WARN] Memory is SLIGHTLY useful - marginal gain")
    else:
        print("[FAIL] Memory is NOISE - no gain or degradation")

    # Save results
    output = {
        'baseline': baseline_results,
        'memory_only': memory_results,
        'full_system': full_results,
    }

    with open('retrieval_test_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to: retrieval_test_results.json")


if __name__ == "__main__":
    main()
