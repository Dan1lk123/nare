"""Autopilot — Autonomous task completion loop."""

from nare.utils.logger import get_logger
import time
from typing import Optional, Dict, Any

log = get_logger("nare.cli.autopilot")


class AutopilotAgent:

    def __init__(self, session, max_iterations: int = 5):
        self.session = session
        self.max_iterations = max_iterations
        self.iteration = 0

    def run(self, task: str, thinking_display=None) -> Dict[str, Any]:
        log.info(f"[Autopilot] Starting task: {task}")

        self.iteration = 0
        last_error = None
        accumulated_context = []

        while self.iteration < self.max_iterations:
            self.iteration += 1

            if thinking_display:
                thinking_display.stream_token(f"\n[Autopilot] Iteration {self.iteration}/{self.max_iterations}\n")

            enriched_task = self._build_query(task, accumulated_context, last_error)

            result = self.session.solve(enriched_task, thinking_display=thinking_display)

            verification = self._verify_result(result, task)

            if verification["success"]:
                log.info(f"[Autopilot] Task completed successfully in {self.iteration} iterations")
                result["autopilot_iterations"] = self.iteration
                result["autopilot_success"] = True
                return result

            last_error = verification["error"]
            accumulated_context.append({
                "iteration": self.iteration,
                "attempt": result.get("final_answer", ""),
                "error": last_error,
            })

            if thinking_display:
                thinking_display.stream_token(f"[Autopilot] Verification failed: {last_error}\n")
                thinking_display.stream_token(f"[Autopilot] Retrying...\n")

            log.warning(f"[Autopilot] Iteration {self.iteration} failed: {last_error}")

        log.error(f"[Autopilot] Failed to complete task after {self.max_iterations} iterations")
        return {
            "final_answer": f"Failed to complete task after {self.max_iterations} attempts. Last error: {last_error}",
            "autopilot_iterations": self.iteration,
            "autopilot_success": False,
            "route_decision": "AUTOPILOT_FAILED",
        }

    def _build_query(self, original_task: str, context: list, last_error: Optional[str]) -> str:
        if not context:
            return (
                f"{original_task}\n\n"
                "IMPORTANT: After completing the task, verify your work by:\n"
                "1. Running any tests if applicable\n"
                "2. Checking that files were created/modified correctly\n"
                "3. Confirming the output matches requirements\n\n"
                "Provide clear verification steps in your response."
            )

        query = f"{original_task}\n\nPREVIOUS ATTEMPTS:\n"
        for ctx in context[-2:]:
            query += f"\nIteration {ctx['iteration']}:\n"
            query += f"Error: {ctx['error']}\n"

        query += (
            f"\nCURRENT ERROR: {last_error}\n\n"
            "Analyze what went wrong and provide a corrected solution.\n"
            "Focus on fixing the specific error mentioned above.\n"
            "Verify your solution before responding."
        )

        return query

    def _verify_result(self, result: Dict[str, Any], task: str) -> Dict[str, Any]:
        answer = result.get("final_answer", "")

        error_indicators = [
            "Error:",
            "Failed:",
            "Exception:",
            "could not",
            "unable to",
            "not found",
        ]

        for indicator in error_indicators:
            if indicator in answer:
                return {
                    "success": False,
                    "error": f"Execution error detected: {indicator}"
                }

        if any(keyword in task.lower() for keyword in ["create", "add"]):
            if "Created" not in answer and "Created:" not in answer:
                return {
                    "success": False,
                    "error": "No file creation confirmation found"
                }

        route = result.get("route_decision", "")
        if route == "SLOW":
            candidates = result.get("generated_candidates", [])
            if candidates:
                score = candidates[0].get("final_score", 0)
                if score < 0.5:
                    return {
                        "success": False,
                        "error": f"Low confidence solution (score: {score:.2f})"
                    }

        return {"success": True, "error": None}
