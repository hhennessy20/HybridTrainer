"""Hybrid RL + Local LLM Recommendation Policy for HybridTrainer.

Fuses the sequential decision-making of Reinforcement Learning (Tabular RL or Deep Q-Network)
with the natural language generation and contextual synthesis of a local LLM (Qwen 3.5 0.8B via Ollama).
"""

import json
import re
import time
import urllib.request
import sys
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Baseline.environment.schemas import UserState, validate_workout_action, ValidationResult
from Baseline.policies.base import BasePolicy
from Baseline.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS


class HybridPolicy(BasePolicy):
    """Hybrid recommendation policy combining an RL decision engine with local LLM synthesis."""

    def __init__(
        self,
        base_rl_policy: BasePolicy,
        name: Optional[str] = None,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = OLLAMA_TIMEOUT_SECONDS,
        use_llm: bool = True,
    ):
        policy_name = name or f"Hybrid_{base_rl_policy.name}_LLM"
        super().__init__(name=policy_name)
        self.rl_policy = base_rl_policy
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.use_llm = use_llm

        # Latency diagnostics
        self.rl_latency_ms: float = 0.0
        self.llm_latency_ms: float = 0.0
        self.last_latency_ms: float = 0.0
        self.last_raw_prompt: str = ""
        self.last_raw_response: str = ""
        self.active_model: str = model

    def is_llm_available(self) -> Tuple[bool, str]:
        """Check if local Ollama server is reachable using proxy-free connection."""
        tags_url = f"{self.base_url}/api/tags"
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(tags_url, headers={"User-Agent": "HybridTrainer/1.0"})
            with opener.open(req, timeout=5) as response:
                if response.status != 200:
                    return False, f"Ollama HTTP {response.status}"
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                if self.model in models:
                    self.active_model = self.model
                    return True, f"Found {self.model}"
                prefix = self.model.split(":")[0]
                for m in models:
                    if m.startswith(prefix):
                        self.active_model = m
                        return True, f"Matched {m}"
                return False, f"Model {self.model} not found in {models}"
        except Exception as e:
            return False, f"Ollama unavailable: {e}"

    def build_synthesis_prompt(self, state: UserState, rl_action: Dict[str, Any]) -> str:
        """Construct structured prompt for LLM constraint adaptation and fine-tuning."""
        from Baseline.environment.exercises import get_exercises_for_equipment
        avail_exercises = get_exercises_for_equipment(state.available_equipment)

        # Filter out high-impact exercises if restricted
        restrictions_set = {r.lower().strip() for r in state.restrictions}
        if "low_impact_preferred" in restrictions_set:
            avail_exercises = [
                ex for ex in avail_exercises 
                if ex.name.lower() not in ["outdoor running", "jump rope", "burpees"]
            ]

        ex_summary = ", ".join([f"{ex.name} ({ex.required_equipment})" for ex in avail_exercises[:8]])

        macro_type = rl_action.get("workout_type", "general")
        base_intensity = float(rl_action.get("target_intensity", 5.0))
        duration = int(rl_action.get("estimated_duration_min", state.available_time_min))
        volume = str(rl_action.get("volume", "moderate"))

        fit_shift = {"beginner": 0.5, "intermediate": 0.0, "advanced": -0.5}.get(state.fitness_level, 0.0)
        ideal_target_int = max(1.0, min(10.0, round(state.target_rpe - 1.5 * state.fatigue - 0.5 * state.stress - fit_shift, 1)))

        is_exhausted = state.fatigue >= 0.70

        prompt = f"""You are an expert AI personal trainer.
A trained reinforcement learning policy has prescribed the macro-strategy:
- RL Recommended Category: {macro_type}
- RL Intensity Guideline: {base_intensity:.1f}
- Prescribed Duration: {duration} minutes (user available limit: {state.available_time_min} minutes)
- Training Volume: {volume}
- Recovery Mandate: {'STRICT: User is exhausted (fatigue >= 0.70). Must prescribe recovery or flexibility.' if is_exhausted else 'Normal training allowed.'}

USER CONSTRAINTS & PHYSIOLOGICAL STATE:
- Fitness Level: {state.fitness_level}
- Primary Goal: {state.goal}
- Current Fatigue: {state.fatigue:.2f} (0=fresh, 1=exhausted)
- Current Stress: {state.stress:.2f} (0=calm, 1=high stress)
- Target Borg RPE: {state.target_rpe:.1f} (Scale 1-10)
- Available Equipment: {state.available_equipment}
- Preferred Categories: {state.preferred_categories}
- Dispreferred Categories: {state.dispreferred_categories}
- Restrictions: {state.restrictions}

ALLOWED EXERCISES MATCHING USER EQUIPMENT & RESTRICTIONS:
{ex_summary}

YOUR TASK (Constraint Adaptation & Micro-Personalization):
1. Workout Category: {'Must be "recovery" or "flexibility" due to high fatigue.' if is_exhausted else f'Adapt category to align with user preferred categories {state.preferred_categories} or goal ({state.goal}), strictly avoiding {state.dispreferred_categories}. Choose from: endurance, strength, hiit, flexibility, recovery.'}
2. Intensity Calibration: Tune target_intensity close to {ideal_target_int} so the user's actual perceived exertion matches Target Borg RPE {state.target_rpe:.1f}.
3. Exercise Selection: Select 2 exercises strictly from the allowed list matching equipment and restrictions.
4. Duration: Set estimated_duration_min <= {state.available_time_min} minutes.
5. Rationale: Write a concise 1-sentence rationale explaining the personalized adaptation.

RESPONSE FORMAT:
Respond with ONLY a single valid JSON object. No conversational intro, no markdown code fence.
Schema:
{{
  "workout_type": "endurance" | "strength" | "hiit" | "flexibility" | "recovery",
  "target_intensity": {ideal_target_int},
  "estimated_duration_min": {min(45, state.available_time_min)},
  "volume": "{volume}",
  "exercises": [
    {{"name": "<exercise name>", "category": "<category>", "equipment": "<equipment>", "sets": 3, "reps": 10, "duration_min": null, "coaching_cue": "<cue>"}}
  ],
  "rationale": "<1-sentence rationale>"
}}"""
        return prompt

    def call_ollama(self, prompt: str) -> str:
        """Call Ollama generate endpoint using proxy-free handler."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.active_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 450,
            },
        }
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "") or data.get("thinking", "")

    @staticmethod
    def extract_json(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Safely extract valid JSON dictionary from LLM response."""
        text = raw_text.strip()
        if not text:
            return None, "Empty response"

        # Regex search for JSON markdown block or curly braces
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if m:
            try:
                return json.loads(m.group(1)), None
            except Exception:
                pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1]), None
            except Exception as e:
                return None, str(e)

        return None, "No JSON found"

    def _adapt_fallback(self, state: UserState, rl_action: Dict[str, Any]) -> Dict[str, Any]:
        """Grounded fallback adapting RL macro-decision to user target RPE, preferences, and restrictions."""
        adapted = dict(rl_action)
        fit_shift = {"beginner": 0.5, "intermediate": 0.0, "advanced": -0.5}.get(state.fitness_level, 0.0)

        # 1. Modality adaptation respecting RL fatigue safety
        dispref = {d.lower().strip() for d in state.dispreferred_categories}
        pref = [p.lower().strip() for p in state.preferred_categories if p.lower().strip() not in dispref]

        if state.fatigue >= 0.70:
            adapted["workout_type"] = "recovery"
        elif pref:
            adapted["workout_type"] = pref[0]
        elif state.goal in ["endurance", "strength", "hiit"] and state.goal not in dispref:
            adapted["workout_type"] = state.goal
        else:
            adapted["workout_type"] = rl_action.get("workout_type", "general")

        # 2. Precise intensity calibration to match target RPE
        ideal_int = state.target_rpe - 1.5 * state.fatigue - 0.5 * state.stress - fit_shift
        tuned_int = max(1.0, min(10.0, round(ideal_int, 1)))
        adapted["target_intensity"] = tuned_int

        # 3. Exercise selection matching equipment and restrictions
        from Baseline.environment.exercises import get_exercises_for_equipment
        avail_exercises = get_exercises_for_equipment(state.available_equipment)
        restrictions_set = {r.lower().strip() for r in state.restrictions}
        if "low_impact_preferred" in restrictions_set:
            avail_exercises = [
                ex for ex in avail_exercises 
                if ex.name.lower() not in ["outdoor running", "jump rope", "burpees"]
            ]

        w_type = adapted["workout_type"]
        matching = [ex for ex in avail_exercises if ex.category.startswith(w_type) or w_type.startswith(ex.category)]
        if not matching:
            matching = avail_exercises

        dur = min(state.available_time_min, int(rl_action.get("estimated_duration_min", state.available_time_min)))
        adapted["estimated_duration_min"] = dur

        chosen = []
        for ex in matching[:2]:
            item = {
                "name": ex.name,
                "category": ex.category,
                "equipment": ex.required_equipment,
            }
            if ex.default_duration_min is not None:
                item["duration_min"] = max(5, dur // max(1, len(matching[:2])))
            else:
                item["sets"] = ex.default_sets or 3
                item["reps"] = ex.default_reps or 10
            chosen.append(item)
        adapted["exercises"] = chosen
        adapted["rationale"] = (
            f"Adapted {w_type} session fine-tuned to target RPE {state.target_rpe:.1f} (intensity {tuned_int}) "
            f"matching preferences {state.preferred_categories} and current fatigue {state.fatigue:.2f}."
        )
        return adapted

    def recommend(self, state: UserState) -> Dict[str, Any]:
        """Generate hybrid recommendation: RL decision + local LLM synthesis."""
        # 1. RL Decision Pass
        t_rl0 = time.perf_counter()
        rl_action = self.rl_policy.recommend(state)
        self.rl_latency_ms = (time.perf_counter() - t_rl0) * 1000.0

        if not self.use_llm:
            self.last_latency_ms = self.rl_latency_ms
            return rl_action

        # 2. Local LLM Synthesis Pass
        avail, _ = self.is_llm_available()
        if not avail:
            adapted = self._adapt_fallback(state, rl_action)
            self.last_latency_ms = self.rl_latency_ms
            return adapted

        prompt = self.build_synthesis_prompt(state, rl_action)
        self.last_raw_prompt = prompt

        t_llm0 = time.perf_counter()
        try:
            raw_resp = self.call_ollama(prompt)
            self.last_raw_response = raw_resp
            parsed, err = self.extract_json(raw_resp)

            if parsed is not None:
                # Validate synthesized action against constraints
                val = validate_workout_action(parsed, state)
                if val.is_valid:
                    self.llm_latency_ms = (time.perf_counter() - t_llm0) * 1000.0
                    self.last_latency_ms = self.rl_latency_ms + self.llm_latency_ms
                    return val.normalized_action or parsed
        except Exception:
            pass

        # If LLM failed or generated invalid constraints, fall back safely to adapted RL action
        self.llm_latency_ms = (time.perf_counter() - t_llm0) * 1000.0
        self.last_latency_ms = self.rl_latency_ms + self.llm_latency_ms
        return self._adapt_fallback(state, rl_action)
