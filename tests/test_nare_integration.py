"""Test NARE integration with AgentLoop."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from nare.cli.session import NareSession

def test_nare_integration():
    """Test that AgentLoop now uses NARE memory and routing."""

    # Create session in test directory
    session = NareSession(repo_path=os.path.dirname(__file__))

    # Test query that should be saved to memory
    query = "What is 2 + 2?"

    print("Testing NARE integration...")
    print(f"Query: {query}")

    # First run - should go through AgentLoop and save to memory
    result1 = session.solve(query)
    print(f"\nFirst run:")
    print(f"  Route: {result1.get('route_decision')}")
    print(f"  Answer: {result1.get('final_answer', '')[:100]}")
    print(f"  NARE hit: {result1.get('_nare_hit', False)}")

    # Check memory
    episodes = session.agent.memory.episodic_memory
    print(f"\nMemory episodes: {len(episodes)}")

    # Second run - should potentially hit NARE cache
    result2 = session.solve(query)
    print(f"\nSecond run:")
    print(f"  Route: {result2.get('route_decision')}")
    print(f"  NARE hit: {result2.get('_nare_hit', False)}")

    print("\n✓ NARE integration test completed")

if __name__ == "__main__":
    test_nare_integration()
