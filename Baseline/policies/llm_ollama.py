"""LLM-only baseline policy using local Ollama and Qwen 3.5.

Makes real local inference calls to Ollama HTTP API with defensive JSON parsing,
shared validation, latency tracking, and honest failure reporting.
"""

import json
import re
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Tuple, Optional, List
from Baseline.environment.schemas import UserState, validate_workout_action, ValidationResult
from Baseline.environment.exercises import get_exercises_for_equipment
from Baseline.policies.base import BasePolicy
from Baseline.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
)

class OllamaLLMPolicy(BasePolicy):
    def __init__(
        self,
        name: str = "LLM_Ollama",
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = OLLAMA_TIMEOUT_SECONDS,
    ):
        super().__init__(name=name)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_latency_ms: float = 0.0
        self.last_raw_prompt: str = ""
        self.last_raw_response: str = ""
        self.last_parse_failed: bool = False
        self.repair_attempted: bool = False
        self.active_model: str = model

    def is_available(self) -> Tuple[bool, str]:
        """Check if Ollama server is reachable and find the matching model."""
        tags_url = f"{self.base_url}/api/tags"
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(tags_url, headers={"User-Agent": "HybridTrainer/1.0"})
            with opener.open(req, timeout=5) as response:
                if response.status != 200:
                    return False, f"Ollama returned HTTP {response.status}"
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                
                # Check preferred model first
                if self.model in models:
                    self.active_model = self.model
                    return True, f"Model '{self.model}' is available."

                # Check family matches (e.g. qwen3.5:latest or qwen3.5:0.8b)
                prefix = self.model.split(":")[0]
                for m in models:
                    if m.startswith(prefix):
                        self.active_model = m
                        return True, f"Model '{self.model}' resolved to available variant '{m}'."

                return False, f"Model '{self.model}' not found in installed models: {models}"
        except urllib.error.URLError as e:
            return False, f"Cannot connect to Ollama at {self.base_url}: {e}"
        except Exception as e:
            return False, f"Ollama check failed: {e}"

    def build_prompt(self, state: UserState) -> str:
        """Construct structured prompt for LLM workout recommendation."""
        avail_exercises = get_exercises_for_equipment(state.available_equipment)
        ex_summary = ", ".join([f"{ex.name} ({ex.required_equipment})" for ex in avail_exercises[:8]])

        prompt = f"""You are an expert personal trainer. Given the following user profile and constraints, recommend a tailored workout in JSON.

USER STATE AND CONSTRAINTS:
- Fitness level: {state.fitness_level}
- Goal: {state.goal}
- Available workout time: {state.available_time_min} minutes (STRICT: estimated_duration_min must NOT exceed {state.available_time_min})
- Available equipment: {state.available_equipment} (STRICT: only use equipment in this list or bodyweight)
- Current fatigue: {state.fatigue} (0.0 to 1.0, where >=0.70 means exhausted/needs recovery)
- Current stress: {state.stress} (0.0 to 1.0)
- Target RPE: {state.target_rpe} (Borg scale 1-10)
- Preferred categories: {state.preferred_categories}
- Dispreferred categories: {state.dispreferred_categories}

ALLOWED EXERCISES MATCHING USER EQUIPMENT:
{ex_summary}

RESPONSE FORMAT:
Respond with ONLY a single JSON object. No conversational text, no markdown code fence, no explanations outside JSON.
Schema:
{{
  "workout_type": "endurance" | "strength" | "hiit" | "flexibility" | "recovery",
  "target_intensity": {state.target_rpe},
  "estimated_duration_min": {min(45, state.available_time_min)},
  "volume": "low" | "moderate" | "high",
  "exercises": [
    {{"name": "<exercise name>", "category": "<category>", "equipment": "<equipment>", "sets": 3, "reps": 10, "duration_min": 15}}
  ],
  "rationale": "<brief 1-sentence rationale>"
}}"""
        return prompt

    def call_ollama(self, prompt: str) -> str:
        """Execute HTTP POST call to Ollama generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.active_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 450,
            },
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=self.timeout) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            # Prefer response, with fallback to thinking if response is empty
            resp_text = resp_data.get("response", "")
            if not resp_text.strip():
                resp_text = resp_data.get("thinking", "")
            return resp_text

    @staticmethod
    def extract_json(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Defensively extract JSON dictionary from raw model text."""
        text = raw_text.strip()
        if not text:
            return None, "Empty response from LLM."

        # Strip markdown fences if present
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if m:
            candidate = m.group(1).strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed, None
            except json.JSONDecodeError:
                pass

        # Try to find JSON object containing workout_type
        m_obj = re.search(r"(\{[\s\S]*?\"workout_type\"[\s\S]*?\})", text)
        if m_obj:
            try:
                parsed = json.loads(m_obj.group(1))
                if isinstance(parsed, dict):
                    return parsed, None
            except json.JSONDecodeError:
                pass

        # Fallback: slice from first '{' to last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed, None
            except json.JSONDecodeError as err:
                return None, f"JSON decode error: {err}"

        return None, "No valid JSON object could be extracted."

    def recommend(self, state: UserState) -> Dict[str, Any]:
        """Generate recommendation via Ollama call, with defensive parsing and 1 repair attempt."""
        prompt = self.build_prompt(state)
        self.last_raw_prompt = prompt
        self.repair_attempted = False
        self.last_parse_failed = False

        t0 = time.perf_counter()
        try:
            raw_resp = self.call_ollama(prompt)
        except Exception as e:
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            self.last_raw_response = f"ERROR: {e}"
            self.last_parse_failed = True
            # Return explicit failure payload (do not silently substitute another policy)
            return {
                "workout_type": "invalid",
                "target_intensity": 0.0,
                "estimated_duration_min": 0,
                "volume": "invalid",
                "exercises": [],
                "rationale": f"LLM inference call failed: {e}",
            }

        self.last_raw_response = raw_resp
        parsed, parse_err = self.extract_json(raw_resp)

        # If parsing or basic validation fails, attempt 1 repair retry
        repair_needed = False
        repair_reason = ""
        if parsed is None:
            repair_needed = True
            repair_reason = f"JSON parse failure: {parse_err}"
        else:
            val_initial = validate_workout_action(parsed, state)
            if not val_initial.is_valid:
                repair_needed = True
                repair_reason = f"Constraint violations: {'; '.join(val_initial.violations)}"

        if repair_needed:
            self.repair_attempted = True
            repair_prompt = (
                f"{prompt}\n\n"
                f"NOTICE: Your previous response failed with the following errors:\n"
                f"{repair_reason}\n\n"
                f"Fix these errors immediately. Output ONLY a valid corrected JSON object."
            )
            try:
                raw_repair = self.call_ollama(repair_prompt)
                parsed_repair, parse_err_repair = self.extract_json(raw_repair)
                if parsed_repair is not None:
                    parsed = parsed_repair
                    self.last_raw_response = raw_repair
            except Exception:
                pass  # Keep initial parsed or None

        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        if parsed is None:
            self.last_parse_failed = True
            return {
                "workout_type": "invalid",
                "target_intensity": 0.0,
                "estimated_duration_min": 0,
                "volume": "invalid",
                "exercises": [],
                "rationale": f"LLM parsing failed after repair attempt: {parse_err}",
            }

        return parsed
