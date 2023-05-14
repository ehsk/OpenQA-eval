import logging
import os
import requests
import time
import numpy as np
from typing import Optional

import openai

logger = logging.getLogger("eval")


class OpenAIProxy:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_attempts: int = 5,
        sleep_time: int = 10,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_attempts = max_attempts
        self.sleep_time = sleep_time

        assert "OPENAI_API_KEY" in os.environ
        openai.api_key = os.environ["SEMIR_OPENAI_KEY"]

        self.call_times = []
        self.num_errors = 0
        self.total_tokens = []

    def _is_conversational(self):
        return self.model_name in ("gpt-4", "gpt-3.5-turbo")

    def __call__(self, text: str, instruction: Optional[str] = None):
        if self._is_conversational():
            prompt = []
            if instruction:
                prompt.append(
                    {
                        "role": "system",
                        "content": instruction,
                    }
                )
            prompt.append(
                {
                    "role": "user",
                    "content": text,
                }
            )
        else:
            if instruction:
                prompt = f"{instruction}\n\n{text}"
            else:
                prompt = text

        for attempt in range(1, self.max_attempts + 1):
            try:
                s = time.time()
                if self._is_conversational():
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                    )
                else:
                    response = openai.Completion.create(
                        model=self.model_name,
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                    )
                e = time.time()

                self.call_times.append(e - s)
                self.total_tokens.append(response["usage"]["total_tokens"])
                if self._is_conversational():
                    return response["choices"][0]["message"]["content"].strip()
                else:
                    return response["choices"][0]["text"].strip()
            except (openai.error.ServiceUnavailableError, openai.error.RateLimitError) as e:
                logger.warning(f"[Attempt {attempt}] rate limit/availability error encountered: {e}")
                self.num_errors += 1
                time.sleep(self.sleep_time + 5 * attempt)
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"[Attempt {attempt}] connection error encountered: {e}")
                self.num_errors += 1
                time.sleep(self.sleep_time + 5 * attempt)

        raise ValueError(f"API call failed after {self.max_attempts} attempts")

    def get_stats(self, reset: bool = True):
        num_calls = len(self.call_times)
        avg_call_time = np.mean(self.call_times)
        std_call_time = np.std(self.call_times)

        num_errors = self.num_errors

        total_tokens = np.sum(self.total_tokens)
        avg_tokens = np.mean(self.total_tokens)

        if reset:
            self.call_times = []
            self.num_errors = 0
            self.total_tokens = []

        return {
            "mean call time": avg_call_time,
            "std call time": std_call_time,
            "num calls": num_calls,
            "num errors": num_errors,
            "total #tokens": total_tokens,
            "avg #tokens": avg_tokens,
        }
