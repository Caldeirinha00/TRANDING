import json
import asyncio
import logging
from openai import AsyncOpenAI
from typing import Dict, Any, Optional, List

from core.config import CONFIG
from risk.kill_switch import trigger_kill_switch

logger = logging.getLogger(__name__)

class MistralClient:
    """Async client for Mistral API with Key Rotation, retry logic and model fallback."""
    
    def __init__(self):
        self.max_retries = 3
        self.timeout = 30.0
        self.consecutive_failures = 0
        
        # Load and parse API keys
        raw_keys = getattr(CONFIG, "MISTRAL_API_KEYS", "")
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip() and k.strip() != "your_mistral_api_key_here"]
        
        if not self.api_keys:
            logger.warning("No Mistral API keys configured in MISTRAL_API_KEYS. Calls will fail.")
            self.api_keys = [""]
            
        self.current_key_idx = 0
        self.primary_model = getattr(CONFIG, "LLM_MODEL", "ministral-8b-2512")
        self.fallback_model = getattr(CONFIG, "LLM_FALLBACK_MODEL", "ministral-3b-2512")
        
        # We will create the client dynamically per request to allow key rotation
        logger.info(f"MistralClient initialized with {len(self.api_keys)} keys. Primary model: {self.primary_model}")

    def _get_current_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=self.api_keys[self.current_key_idx],
            timeout=self.timeout
        )

    def _rotate_key(self):
        """Rotate to the next API key in the list."""
        old_idx = self.current_key_idx
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        logger.warning(f"[MISTRAL CLIENT] Key {old_idx + 1} exhausted/failed. Rotating to Key {self.current_key_idx + 1} of {len(self.api_keys)}.")

    async def _call_api(self, model: str, prompt: str, system_prompt: str) -> Optional[str]:
        """Make a single API call to Mistral."""
        client = self._get_current_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=1500
        )
        return response.choices[0].message.content

    def _parse_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling markdown wrappers and truncated brackets."""
        if not content:
            return None
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt simple auto-repair for truncated JSON responses (e.g. unclosed quotes or braces)
            try:
                repaired = cleaned
                # Close unclosed strings
                if repaired.count('"') % 2 != 0:
                    repaired += '"'
                # Close unclosed brackets
                open_braces = repaired.count('{') - repaired.count('}')
                if open_braces > 0:
                    repaired += '}' * open_braces
                return json.loads(repaired)
            except Exception:
                pass

            logger.error(f"Failed to parse JSON. Raw: {content[:200]}")
            return None

    async def get_json_analysis(
        self, 
        prompt: str, 
        system_prompt: str = "You are a financial AI. Respond ONLY with valid JSON.",
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Calls API with retry + exponential backoff and API Key Rotation.
        Implements Circuit Breaker if all keys fail.
        """
        target_model = model or self.primary_model
        models_to_try = [target_model]
        if target_model != self.fallback_model:
            models_to_try.append(self.fallback_model)
        
        for current_model in models_to_try:
            # We allow (max_retries * number of keys) total attempts to cycle through all keys
            max_attempts = self.max_retries * len(self.api_keys)
            
            for attempt in range(1, max_attempts + 1):
                try:
                    content = await self._call_api(current_model, prompt, system_prompt)
                    parsed = self._parse_json(content)
                    if parsed is not None:
                        self.consecutive_failures = 0 # Reset circuit breaker
                        return parsed
                    
                    logger.warning(f"Model {current_model} returned unparseable JSON (attempt {attempt})")
                    
                except Exception as e:
                    error_str = str(e).lower()
                    logger.warning(f"LLM API error (model={current_model}, attempt {attempt}/{max_attempts}): {e}")
                    
                    # If Rate Limit (429) or Quota exceeded, rotate key immediately!
                    if "429" in error_str or "too many requests" in error_str or "quota" in error_str:
                        if len(self.api_keys) > 1:
                            self._rotate_key()
                            # Short sleep to let rotation take effect without aggressive backoff
                            await asyncio.sleep(1)
                            continue
                
                # Standard exponential backoff for other errors (500, timeouts)
                if attempt < max_attempts:
                    wait_time = min(30, 2 ** attempt)  # Cap wait time at 30s
                    await asyncio.sleep(wait_time)
            
            logger.warning(f"All attempts exhausted for {current_model}, trying fallback model...")
        
        logger.error("All models, keys, and retries exhausted.")
        
        self.consecutive_failures += 1
        logger.critical(f"LLM API consecutive failures: {self.consecutive_failures}/3")
        
        if self.consecutive_failures >= 3:
            logger.critical("Circuit Breaker Activated! LLM API failed 3 times. Waiting 60s cooldown...")
            await asyncio.sleep(60)
            
            # Try rotating one last time before final attempt
            if len(self.api_keys) > 1:
                self._rotate_key()
                
            try:
                content = await self._call_api(self.primary_model, prompt, system_prompt)
                parsed = self._parse_json(content)
                if parsed is not None:
                    logger.info("LLM API recovered after 60s cooldown. Resetting circuit breaker.")
                    self.consecutive_failures = 0
                    return parsed
            except Exception as e:
                pass
                
            logger.critical("LLM API still dead after 60s cooldown. TRIGGERING KILL SWITCH.")
            await trigger_kill_switch("LLM API Dead - Keys Exhausted", None)
            
        return None

mistral_client = MistralClient()
