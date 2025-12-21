# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Some LLM inference interface."""

import abc
import base64
import io
import os
import time
from typing import Any, Optional, Tuple
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import answer_types
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
from google.generativeai.types import safety_types
import numpy as np
from PIL import Image
from openai import OpenAI
import requests

ERROR_CALLING_LLM = 'Error calling LLM'


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Converts a numpy array into a byte string for a JPEG image."""
    image = Image.fromarray(image)
    return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format='JPEG')
    # Reset file pointer to start
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    return img_bytes


class LlmWrapper(abc.ABC):
    """Abstract interface for (text only) LLM."""

    @abc.abstractmethod
    def predict(
            self,
            text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        """Calling text-only LLM with a prompt.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output, is_safe, and raw output.
    """


class MultimodalLlmWrapper(abc.ABC):
    """Abstract interface for Multimodal LLM."""

    @abc.abstractmethod
    def predict_mm(
            self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """


SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
}


class GeminiGcpWrapper(LlmWrapper, MultimodalLlmWrapper):
    """Gemini GCP interface."""

    def __init__(
            self,
            model_name: str | None = None,
            max_retry: int = 3,
            temperature: float = 0.0,
            top_p: float = 0.95,
            enable_safety_checks: bool = True,
    ):
        if 'GCP_API_KEY' not in os.environ:
            raise RuntimeError('GCP API key not set.')
        genai.configure(api_key=os.environ['GCP_API_KEY'])
        self.llm = genai.GenerativeModel(
            model_name,
            safety_settings=None
            if enable_safety_checks
            else SAFETY_SETTINGS_BLOCK_NONE,
            generation_config=generation_types.GenerationConfig(
                temperature=temperature, top_p=top_p
            ),
        )
        if max_retry <= 0:
            max_retry = 3
            print('Max_retry must be positive. Reset it to 3')
        self.max_retry = min(max_retry, 5)

    def predict(
            self,
            text_prompt: str,
            enable_safety_checks: bool = True,
            generation_config: generation_types.GenerationConfigType | None = None,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(
            text_prompt, [], enable_safety_checks, generation_config
        )

    def is_safe(self, raw_response):
        try:
            return (
                    raw_response.candidates[0].finish_reason
                    != answer_types.FinishReason.SAFETY
            )
        except Exception:  # pylint: disable=broad-exception-caught
            #  Assume safe if the response is None or doesn't have candidates.
            return True

    def predict_mm(
            self,
            text_prompt: str,
            images: list[np.ndarray],
            enable_safety_checks: bool = True,
            generation_config: generation_types.GenerationConfigType | None = None,
    ) -> tuple[str, Optional[bool], Any]:
        counter = self.max_retry
        retry_delay = 1.0
        output = None
        while counter > 0:
            try:
                output = self.llm.generate_content(
                    [text_prompt] + [Image.fromarray(image) for image in images],
                    safety_settings=None
                    if enable_safety_checks
                    else SAFETY_SETTINGS_BLOCK_NONE,
                    generation_config=generation_config,
                )
                return output.text, True, output
            except Exception as e:  # pylint: disable=broad-exception-caught
                counter -= 1
                print('Error calling LLM, will retry in {retry_delay} seconds')
                print(e)
                if counter > 0:
                    # Expo backoff
                    time.sleep(retry_delay)
                    retry_delay *= 2

        if (output is not None) and (not self.is_safe(output)):
            return ERROR_CALLING_LLM, False, output
        return ERROR_CALLING_LLM, None, None

    def generate(
            self,
            contents: (
                    content_types.ContentsType | list[str | np.ndarray | Image.Image]
            ),
            safety_settings: safety_types.SafetySettingOptions | None = None,
            generation_config: generation_types.GenerationConfigType | None = None,
    ) -> tuple[str, Any]:
        """Exposes the generate_content API.

    Args:
      contents: The input to the LLM.
      safety_settings: Safety settings.
      generation_config: Generation config.

    Returns:
      The output text and the raw response.
    Raises:
      RuntimeError:
    """
        counter = self.max_retry
        retry_delay = 1.0
        response = None
        if isinstance(contents, list):
            contents = self.convert_content(contents)
        while counter > 0:
            try:
                response = self.llm.generate_content(
                    contents=contents,
                    safety_settings=safety_settings,
                    generation_config=generation_config,
                )
                return response.text, response
            except Exception as e:  # pylint: disable=broad-exception-caught
                counter -= 1
                print('Error calling LLM, will retry in {retry_delay} seconds')
                print(e)
                if counter > 0:
                    # Expo backoff
                    time.sleep(retry_delay)
                    retry_delay *= 2
        raise RuntimeError(f'Error calling LLM. {response}.')

    def convert_content(
            self,
            contents: list[str | np.ndarray | Image.Image],
    ) -> content_types.ContentsType:
        """Converts a list of contents to a ContentsType."""
        converted = []
        for item in contents:
            if isinstance(item, str):
                converted.append(item)
            elif isinstance(item, np.ndarray):
                converted.append(Image.fromarray(item))
            elif isinstance(item, Image.Image):
                converted.append(item)
        return converted


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
    """OpenAI GPT4 wrapper.

  Attributes:
    openai_api_key: The class gets the OpenAI api key either explicitly, or
      through env variable in which case just leave this empty.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: GPT model to use based on if it is multimodal.
  """

    RETRY_WAITING_SECONDS = 20

    def __init__(
            self,
            model_name: str,
            max_retry: int = 3,
            temperature: float = 0.0,
    ):
        if 'OPENAI_API_KEY' not in os.environ:
            raise RuntimeError('OpenAI API key not set.')
        self.openai_api_key = os.environ['OPENAI_API_KEY']
        if max_retry <= 0:
            max_retry = 3
            print('Max_retry must be positive. Reset it to 3')
        self.max_retry = min(max_retry, 5)
        self.temperature = temperature
        self.model = model_name

    @classmethod
    def encode_image(cls, image: np.ndarray) -> str:
        return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

    def predict(
            self,
            text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(
            self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.openai_api_key}',
        }

        payload = {
            'model': self.model,
            'temperature': self.temperature,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': text_prompt},
                ],
            }],
            'max_tokens': 1000,
        }

        # Gpt-4v supports multiple images, just need to insert them in the content
        # list.
        for image in images:
            payload['messages'][0]['content'].append({
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
                },
            })

        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        while counter > 0:
            try:
                response = requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers=headers,
                    json=payload,
                )
                if response.ok and 'choices' in response.json():
                    return (
                        response.json()['choices'][0]['message']['content'],
                        None,
                        response,
                    )
                print(
                    'Error calling OpenAI API with error message: '
                    + response.json()['error']['message']
                )
                time.sleep(wait_seconds)
                wait_seconds *= 2
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Want to catch all exceptions happened during LLM calls.
                time.sleep(wait_seconds)
                wait_seconds *= 2
                counter -= 1
                print('Error calling LLM, will retry soon...')
                print(e)
        return ERROR_CALLING_LLM, None, None


class DeepseekWrapper(LlmWrapper, MultimodalLlmWrapper):
    """DeepSeek wrapper using OpenAI SDK interface."""

    RETRY_WAITING_SECONDS = 10

    def __init__(
            self,
            model_name: str = "deepseek-chat",
            max_retry: int = 3,
            temperature: float = 0.0,
    ):
        # api_key = os.getenv("DEEPSEEK_API_KEY")
        # if not api_key:
        #   raise RuntimeError("DeepSeek API key not set.")
        api_key = 'sk-d645153259c2492c91893c07f6e72b7a'
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model_name
        self.temperature = temperature
        self.max_retry = max(1, min(max_retry, 5))

    @classmethod
    def encode_image(cls, image: np.ndarray) -> str:
        return base64.b64encode(array_to_jpeg_bytes(image)).decode("utf-8")

    def predict(self, text_prompt: str):
        return self.predict_mm(text_prompt, [])

    def predict_mm(self, text_prompt: str, images: list[np.ndarray]):
        # DeepSeek (OpenAI compatible) doesn't yet support image input officially
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text_prompt},
        ]
        counter = self.max_retry
        wait = self.RETRY_WAITING_SECONDS

        while counter > 0:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1000,
                    stream=False,
                )
                return response.choices[0].message.content, True, response
            except Exception as e:
                print("Error calling DeepSeek, retrying...", e)
                counter -= 1
                time.sleep(wait)
                wait *= 2

        return "Error calling LLM", None, None


class LlamaCppWrapper(LlmWrapper, MultimodalLlmWrapper):
    """llama.cpp OpenAI-compatible server wrapper (multimodal)."""

    RETRY_WAITING_SECONDS = 2

    def __init__(
            self,
            api_url: str = "http://localhost:8081/v1/chat/completions",
            max_retry: int = 3,
            temperature: float = 0.0,
            max_tokens: int = 200,
    ):
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max(1, min(max_retry, 5))

    @staticmethod
    def encode_image(image: np.ndarray) -> str:
        """Encode numpy image to data:image/jpeg;base64,..."""
        b64 = base64.b64encode(array_to_jpeg_bytes(image)).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def predict(
            self,
            text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        # text-only fallback
        return self.predict_mm(text_prompt, [])

    def predict_mm(
            self,
            text_prompt: str,
            images: list[np.ndarray],
    ) -> tuple[str, Optional[bool], Any]:
        headers = {"Content-Type": "application/json"}

        # 构造 OpenAI-compatible message
        content = [{"type": "text", "text": text_prompt}]

        for image in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": self.encode_image(image)
                },
            })

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        counter = self.max_retry
        wait = self.RETRY_WAITING_SECONDS

        while counter > 0:
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

                if response.status_code != 200:
                    print("[LlamaCppWrapper] Server error:", response.text)
                    counter -= 1
                    time.sleep(wait)
                    wait *= 2
                    continue

                js = response.json()
                return (
                    js["choices"][0]["message"]["content"],
                    True,  # llama.cpp 无 safety signal
                    js,
                )

            except Exception as e:
                print("[LlamaCppWrapper] Error calling llama.cpp:", e)
                counter -= 1
                time.sleep(wait)
                wait *= 2

        return ERROR_CALLING_LLM, None, None


class LlamaCppTextWrapper(LlmWrapper):
    """llama.cpp OpenAI-compatible server wrapper (text-only)."""

    RETRY_WAITING_SECONDS = 2

    def __init__(
            self,
            api_url: str = "http://localhost:8080/v1/chat/completions",
            max_retry: int = 3,
            temperature: float = 0.0,
            max_tokens: int = 512,
    ):
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max(1, min(max_retry, 5))

    def predict(self, text_prompt: str) -> Tuple[str, Optional[bool], Any]:
        """
        Return:
          - content: model output string
          - safety_flag: None (llama.cpp server does not provide one)
          - raw: parsed JSON response, or None on failure
        """
        headers = {"Content-Type": "application/json"}

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "Do not output chain-of-thought or reasoning. Only give the final answer."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        counter = self.max_retry
        wait = self.RETRY_WAITING_SECONDS

        while counter > 0:
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

                if response.status_code != 200:
                    print("[LlamaCppTextWrapper] Server error:", response.text)
                    counter -= 1
                    time.sleep(wait)
                    wait *= 2
                    continue

                js = response.json()
                return (
                    js["choices"][0]["message"]["content"],
                    True,
                    js,
                )

            except Exception as e:
                print("[LlamaCppTextWrapper] Error calling llama.cpp:", e)
                counter -= 1
                time.sleep(wait)
                wait *= 2

        return ERROR_CALLING_LLM, None, None
