"""Simple test to verify NARE integration without API calls."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all NARE components import correctly."""
    print("Testing imports...")

    from nare.cli.session import NareSession
    print("[OK] NareSession")

    from nare.core.agent import NAREProductionAgent
    print("[OK] NAREProductionAgent")

    from nare.core.router import ReasoningRouter
    print("[OK] ReasoningRouter")

    from nare.core.evolution import EvolutionEngine
    print("[OK] EvolutionEngine")

    from nare.memory.memory import MemorySystem
    print("[OK] MemorySystem")

    print("\n[OK] All imports successful")

def test_session_structure():
    """Test that NareSession has NARE integration."""
    print("\nTesting session structure...")

    from nare.cli.session import NareSession
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        session = NareSession(repo_path=tmpdir)

        # Check that session has agent
        assert hasattr(session, 'agent'), "Session missing agent attribute"
        print("[OK] Session has agent")

        # Check that solve_agentic exists
        assert hasattr(session, 'solve_agentic'), "Session missing solve_agentic"
        print("[OK] Session has solve_agentic method")

        # Initialize agent
        session.init_agent()

        # Check agent components
        assert session.agent is not None, "Agent not initialized"
        print("[OK] Agent initialized")

        assert hasattr(session.agent, 'router'), "Agent missing router"
        print("[OK] Agent has router")

        assert hasattr(session.agent, 'memory'), "Agent missing memory"
        print("[OK] Agent has memory")

        assert hasattr(session.agent, 'evolution'), "Agent missing evolution"
        print("[OK] Agent has evolution")

        print("\n[OK] Session structure correct")

if __name__ == "__main__":
    test_imports()
    test_session_structure()
    print("\n[SUCCESS] All tests passed - NARE integration is connected")
