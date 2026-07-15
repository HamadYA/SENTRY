from __future__ import annotations


class MemoryController:
    def __init__(self, policy: str):
        self.policy = policy

    def admit(self, backend, rescued: bool, mask):
        result = {"policy": self.policy, "attempted": False, "applied": False, "error": None}
        if self.policy != "severe_rescue" or not rescued:
            return result
        result["attempted"] = True
        if mask is None:
            result["error"] = "accepted rescue has no mask"
            return result
        try:
            backend_result = backend.replace_current_memory(mask)
            if backend_result:
                result.update(backend_result)
            if not backend_result or "applied" not in backend_result:
                result["applied"] = True
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        return result
