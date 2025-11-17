from dataclasses import dataclass
import os
import os.path as osp
import base64
import json
import re
from typing import List
import time

import numpy as np
import requests
from kiui.utils import write_image

import torch

import threestudio
from threestudio.utils.base import BaseObject


@threestudio.register("lmm-score")
class LMMVisionScore(BaseObject):
    """
    Generic OpenRouter-based image reward model.

    Works with ANY OpenRouter model that:
      - Is accessible at /api/v1/chat/completions
      - Supports OpenAI-style multimodal messages with `image_url`
        (e.g., Gemini 2.5, GPT-4.1, GPT-4o, future GPT-5 vision models, etc.)

    It takes (prompt, image) and returns a scalar score in [0, 1] indicating
    how well the image matches the prompt.
    """

    @dataclass
    class Config(BaseObject.Config):
        # Any OpenRouter multimodal model, examples:
        #   - "google/gemini-2.5-flash"
        #   - "google/gemini-2.5-pro"
        #   - "openai/gpt-4.1-mini"
        #   - "openai/gpt-4.1"
        #   - future: "openai/gpt-5-vision" (hypothetical)
        model: str = "google/gemini-2.5-pro"

        # Prefer to set via env: OPENROUTER_API_KEY
        api_key: str = os.environ.get("OPENROUTER_API_KEY", "")

        # Where to temporarily dump PNGs for the API call
        temp_folder: str = "outputs/openrouter_temp"

        # OpenRouter Chat Completions endpoint (OpenAI-compatible)
        base_url: str = "https://openrouter.ai/api/v1/chat/completions"

        # System prompt: ask the model to return ONLY a numeric score [0,1].
        system_prompt: str = (
            "You are an expert 3D-aware image evaluator. "
            "Your task is to judge how well a rendered IMAGE matches the intended TEXT PROMPT "
            "for the purpose of generating consistent multi-view 3D assets.\n\n"

            "Focus FIRST on the overall high-level semantics and global appearance: "
            "identity, category, shape, proportions, pose, composition, and major colors.\n"
            "Focus SECOND on fine-grained details: textures, materials, surface patterns, "
            "lighting consistency, and small object attributes.\n\n"

            "Your score must reflect how well the image would serve as one correct view "
            "of a coherent and stable 3D object or scene described by the text.\n"
            "Penalize deviations in structure, shape, or essential features more heavily "
            "than minor texture inaccuracies.\n\n"

            "Output ONLY a SINGLE floating-point number between 0 and 1, where:\n"
            "  • 1 = perfect semantic match and stable 3D-consistent appearance\n"
            "  • 0 = unrelated or useless for 3D generation\n\n"

            "Do NOT provide explanations, words, percentages, or extra formatting. "
            "Return ONLY the numeric score."
        )

    cfg: Config

    def configure(self) -> None:
        threestudio.info(
            f"Loading OpenRouterVisionScore reward model with backend `{self.cfg.model}`."
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Temp folder for intermediate PNGs
        os.makedirs(self.cfg.temp_folder, exist_ok=True)
        self.save_folder = self.cfg.temp_folder

        # HTTP session
        self.session = requests.Session()
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

        # Optional OpenRouter-specific headers (for attribution/analytics)
        headers.setdefault("HTTP-Referer", "https://github.com/your-org/your-project")
        headers.setdefault("X-Title", "DreamDPO-OpenRouter-Vision-Reward")

        self.session.headers.update(headers)

        if not self.cfg.api_key:
            threestudio.warn(
                "OpenRouterVisionScore: OPENROUTER_API_KEY is empty. "
                "Set cfg.api_key or the env var OPENROUTER_API_KEY."
            )

    # -------------------------------------------------------------------------
    # image utils
    # -------------------------------------------------------------------------

    def preprocess(self, image):
        """
        Convert torch tensor in [0,1] or uint8 numpy to HWC float32 [0,1].
        """
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Convert CHW -> HWC if necessary (common for PyTorch tensors)
        if len(image.shape) == 3 and image.shape[0] < image.shape[-1]:
            image = image.transpose(1, 2, 0)

        return image

    def encode_image_to_data_url(self, path: str) -> str:
        """
        Read PNG and convert to data URL for OpenRouter multimodal input.
        """
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # -------------------------------------------------------------------------
    # OpenRouter call + parsing
    # -------------------------------------------------------------------------

    def _build_payload(self, prompt: str, image_data_url: str) -> dict:
        """
        Build an OpenAI-compatible /chat/completions payload for OpenRouter
        with text + image input.
        """
        # OpenRouter uses OpenAI-style multimodal messages:
        # message.content is a list of parts with type "text" or "image_url".
        user_content: List[dict] = [
            {"type": "text", "text": f"TEXT PROMPT:\n{prompt}"},
            {
                "type": "image_url",
                "image_url": {"url": image_data_url},
            },
            {
                "type": "text",
                "text": "Now output ONLY the numeric score between 0 and 1.",
            },
        ]

        payload = {
            "model": self.cfg.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.cfg.system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "max_tokens": 8,
            "temperature": 0.0,
        }
        return payload

    def _parse_score_from_response(self, data: dict) -> float:
        """
        Extract the first float from the model response and clamp to [0, 1].
        """
        message = data["choices"][0]["message"]

        # OpenRouter may return `content` as a string or as a list of parts
        content = message.get("content", "")
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        else:
            text = str(content)

        # print(f"OpenRouterVisionScore response text: {text!r}")

        # Extract first float-like token
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not m:
            raise RuntimeError(
                f"OpenRouterVisionScore: cannot parse numeric score from: {text!r}"
            )
        score = float(m.group())

        # Clamp to [0,1] and heuristically normalize if model uses 0–10 or 0–100
        if score > 1.0:
            if score <= 10.0:
                score /= 10.0
            elif score <= 100.0:
                score /= 100.0

        score = max(0.0, min(1.0, score))
        return score

    def query_model(self, prompt: str, image_path: str) -> float:
        """
        Single-call: prompt + one image -> scalar score in [0,1].
        """
        image_data_url = self.encode_image_to_data_url(image_path)
        payload = self._build_payload(prompt, image_data_url)

        try:
            resp = self.session.post(
                self.cfg.base_url, data=json.dumps(payload), timeout=60
            )
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Helpful error if user picked a non-vision model
            raise RuntimeError(
                f"OpenRouterVisionScore: HTTP error from OpenRouter ({e}). "
                f"Check that model `{self.cfg.model}` supports image inputs."
            ) from e

        data = resp.json()
        return self._parse_score_from_response(data)
    
    def process_single_image(self, i, img, text_input):
        img = self.preprocess(img)
        save_path = osp.join(self.cfg.temp_folder, f"temp{i}.png")
        write_image(save_path, img)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                score = self.query_model(text_input, save_path)
                return i, score
            except Exception as e:
                if attempt == max_retries - 1:
                    threestudio.warn(
                        f"OpenRouter query failed after {max_retries} attempts "
                        f"for image index {i}: {e}"
                    )
                    return i, 0.0

                sleep_time = 0.5 * (2 ** attempt)
                threestudio.warn(
                    f"[Retry {attempt+1}/{max_retries}] OpenRouter error for image {i}: {e}. "
                    f"Retrying in {sleep_time:.1f}s ..."
                )
                time.sleep(sleep_time)

    # -------------------------------------------------------------------------
    # main API used by guidance
    # -------------------------------------------------------------------------

    def __call__(
        self,
        image,
        prompt,
        prompt_utils=None,
        elevation=None,
        azimuth=None,
        camera_distances=None,
    ):
        """
        Args:
            image: [B, C, H, W] torch tensor or equivalent.
            prompt: str, the text prompt used for optimization.

        Returns:
            scores: torch.FloatTensor of shape [B] on self.device.
        """
        scores = [0] * image.shape[0]

        # start = time.time()
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self.process_single_image, i, img, prompt) for i, img in enumerate(image)]
            for future in concurrent.futures.as_completed(futures):
                i, score = future.result()
                scores[i] = score

        # print(f"[LMMVisionScore] Scores: {scores}")
        # print(f"[LMMVisionScore] Time taken for scoring {image.shape[0]} images: {time.time() - start:.2f} seconds")
        scores_tensor = torch.tensor(scores, device=self.device, dtype=torch.float32)
        return scores_tensor
