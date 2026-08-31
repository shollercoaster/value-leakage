"""Hugging Face Inference Providers chat completions API utils.

OpenAI-compatible drop-in endpoint (https://router.huggingface.co/v1), which
Hugging Face's own docs describe as a thin proxy in front of partner
providers (deepinfra, novita, etc.) with no markup on the provider's own
rate. Provider selection happens via a suffix on the model id, not a
separate request field: "Qwen/Qwen3.5-122B-A10B:deepinfra" pins DeepInfra,
":fastest" / ":cheapest" / ":preferred" pick automatically. See
https://huggingface.co/docs/inference-providers.
"""

import asyncio
import os

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

load_dotenv()


def get_huggingface_client() -> AsyncOpenAI:
    """Get an AsyncOpenAI client configured for Hugging Face Inference Providers."""
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("HF_TOKEN not found in .env file!")

    return AsyncOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
    )


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
async def call_api(
    client: AsyncOpenAI,
    model: str,
    messages: list,
    temperature: float = 1.0,
    max_tokens: int = 5000,
    top_p: float = 1.0,
    extra_body: dict | None = None,
    **kwargs,
):
    """Make a chat completion API call to Hugging Face Inference Providers.

    Args:
        client: AsyncOpenAI client configured for Hugging Face.
        model: Model id, optionally with a ":<provider>" or ":fastest" /
            ":cheapest" / ":preferred" suffix (e.g. "Qwen/Qwen3.5-122B-A10B:deepinfra").
        messages: Chat messages.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        top_p: Nucleus sampling threshold.
        extra_body: Extra parameters to pass in the request body (e.g. reasoning
            effort, if the underlying provider supports it).
        **kwargs: Additional parameters forwarded to the API call.

    Returns:
        Full response object from the Hugging Face router.
    """
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        extra_body=extra_body if extra_body else None,
        **kwargs,
    )


async def process_one(
    client: AsyncOpenAI,
    model: str,
    messages: list,
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    max_tokens: int = 5000,
    top_p: float = 1.0,
    extra_body: dict | None = None,
    **kwargs,
):
    """Process a single request with semaphore-based concurrency control."""
    async with semaphore:
        return await call_api(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_body=extra_body,
            **kwargs,
        )


async def process_batch(
    client: AsyncOpenAI,
    model: str,
    messages_list: list,
    temperature: float = 1.0,
    max_tokens: int = 5000,
    max_concurrent: int = 10,
    top_p: float = 1.0,
    extra_body: dict | None = None,
    return_exceptions: bool = False,
    **kwargs,
) -> list:
    """Process all requests concurrently with a semaphore."""
    semaphore = asyncio.Semaphore(max_concurrent)
    coroutines = [
        process_one(
            client=client,
            model=model,
            messages=m,
            semaphore=semaphore,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_body=extra_body,
            **kwargs,
        )
        for m in messages_list
    ]

    if return_exceptions:
        async def wrap_with_progress(coro, pbar):
            try:
                result = await coro
                pbar.update(1)
                return result
            except Exception as e:
                pbar.update(1)
                return e

        from tqdm import tqdm
        pbar = tqdm(total=len(coroutines))
        wrapped = [wrap_with_progress(c, pbar) for c in coroutines]
        results = await asyncio.gather(*wrapped)
        pbar.close()
        return results
    else:
        return await tqdm_asyncio.gather(*coroutines)
