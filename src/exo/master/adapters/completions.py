"""OpenAI Completions API adapter for converting requests/responses."""

import time
from collections.abc import AsyncGenerator
from typing import Any

from exo.shared.types.api import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamResponse,
    CompletionUsage,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    Logprobs,
    LogprobsContentItem,
)
from exo.shared.types.chunks import ErrorChunk, TokenChunk
from exo.shared.types.common import CommandId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams


def completion_request_to_text_generation(
    request: CompletionRequest,
) -> TextGenerationTaskParams:
    input_messages: list[InputMessage] = []
    
    # Handle prompts
    prompts: list[str]
    if isinstance(request.prompt, str):
        prompts = [request.prompt]
    else:
        prompts = request.prompt

    # For now, we only support single prompt for simplicity in this adapter
    # The TextGenerationTaskParams expects a list of InputMessages.
    # For completion, we verify it's usually just "user" content or raw text.
    # We'll treat the prompt as a user message for now, but really we might need 
    # a "system" or "raw" role if we want exact completion without chat formatting.
    # However, TextGenerationTaskParams uses InputMessage which enforces roles.
    # If the underlying model is a base model (not instruct/chat), we might need to be careful.
    # But often "user" role is just treated as input text by the engine if mapped correctly.
    # Let's assume we pass it as 'user' role for now, or maybe we need a 'raw' option?
    # TextGenerationTaskParams has `input: list[InputMessage]`.
    # Let's inspect InputMessage again: role: MessageRole = Literal["user", "assistant", "system", "developer"]
    
    # If we want raw completion, we might need to adjust how TextGenerationTaskParams works 
    # or how the engine interprets it. For now, let's map to 'user' and see.
    # Actually, many base models in exo might expect chat template. 
    # If it's a base model without chat template, 'user' content is usually just passed through.
    
    prompt_content = prompts[0] if prompts else ""
    if request.suffix:
        # FIM support: construct FIM prompt
        # Default to Qwen/DeepSeek style FIM tokens which are common for modern coder models
        # TODO: Detect model type or use tokenizer config to get exact FIM tokens
        PRE = "<|fim_prefix|>"
        SUF = "<|fim_suffix|>"
        MID = "<|fim_middle|>"
        prompt_content = f"{PRE}{prompt_content}{SUF}{request.suffix}{MID}"
        
        # Add default FIM stop tokens if not present
        # Qwen/DeepSeek use these tokens to signal end of FIM or file
        default_stops = [
            "<|file_separator|>",
            "<|fim_prefix|>",
            "<|fim_suffix|>",
            "<|fim_middle|>",
            "<|fim_pad|>",
            "<|repo_name|>",
            "<|file_path|>",
            "<|im_start|>",
            "<|im_end|>",
        ]
        if request.stop is None:
            request.stop = default_stops
        elif isinstance(request.stop, str):
            request.stop = [request.stop] + default_stops
        else:
            request.stop = list(request.stop) + default_stops

    input_messages.append(InputMessage(role="user", content=prompt_content))

    max_tokens = request.max_tokens
    if request.suffix:
        # For autocomplete, we want a tighter limit to ensure snappiness
        if max_tokens is None or max_tokens > 64:
            max_tokens = 64
    else:
        # For general completions from IDE, cap at a sane limit
        if max_tokens is None or max_tokens > 512:
            max_tokens = 512

    return TextGenerationTaskParams(
        model=request.model,
        input=input_messages,
        max_output_tokens=max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        n=request.n,
        stop=request.stop,
        stream=request.stream,
        logprobs=request.logprobs is not None,
        top_logprobs=request.logprobs,
        seed=request.seed,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        is_raw_prompt=True,
    )


def chunk_to_completion_response(
    chunk: TokenChunk, command_id: CommandId, index: int = 0
) -> CompletionResponse:
    """Convert a TokenChunk to a CompletionResponse (non-streaming chunk)."""
    logprobs: Logprobs | None = None
    if chunk.logprob is not None:
        logprobs = Logprobs(
            content=[
                LogprobsContentItem(
                    token=chunk.text,
                    logprob=chunk.logprob,
                    top_logprobs=chunk.top_logprobs or [],
                )
            ]
        )

    return CompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=chunk.model,
        choices=[
            CompletionChoice(
                text=chunk.text,
                index=index,
                logprobs=logprobs,
                finish_reason=chunk.finish_reason,
            )
        ],
    )

def chunk_to_stream_response(
    chunk: TokenChunk, command_id: CommandId, index: int = 0
) -> CompletionStreamResponse:
    """Convert a TokenChunk to a CompletionStreamResponse."""
    logprobs: Logprobs | None = None
    if chunk.logprob is not None:
        logprobs = Logprobs(
            content=[
                LogprobsContentItem(
                    token=chunk.text,
                    logprob=chunk.logprob,
                    top_logprobs=chunk.top_logprobs or [],
                )
            ]
        )

    return CompletionStreamResponse(
        id=command_id,
        created=int(time.time()),
        model=chunk.model,
        choices=[
            CompletionChoice(
                text=chunk.text,
                index=index,
                logprobs=logprobs,
                finish_reason=chunk.finish_reason,
            )
        ],
    )


async def generate_completion_stream(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[ErrorChunk | TokenChunk, None],
) -> AsyncGenerator[str, None]:
    """Generate Completions API streaming events from chunks."""
    async for chunk in chunk_stream:
        if isinstance(chunk, ErrorChunk):
            error_response = ErrorResponse(
                error=ErrorInfo(
                    message=chunk.error_message or "Internal server error",
                    type="InternalServerError",
                    code=500,
                )
            )
            yield f"data: {error_response.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            return

        # We ignore ToolCallChunk for completions endpoint as it's not supported standardly
        if not isinstance(chunk, TokenChunk):
            continue

        chunk_response = chunk_to_stream_response(chunk, command_id)
        yield f"data: {chunk_response.model_dump_json()}\n\n"

        if chunk.finish_reason is not None:
            yield "data: [DONE]\n\n"


async def collect_completion_response(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[ErrorChunk | TokenChunk, None],
) -> CompletionResponse:
    """Collect all token chunks and return a single CompletionResponse."""
    text_parts: list[str] = []
    logprobs_content: list[LogprobsContentItem] = []
    model: str | None = None
    finish_reason: FinishReason | None = None
    error_message: str | None = None
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in chunk_stream:
        if isinstance(chunk, ErrorChunk):
            error_message = chunk.error_message or "Internal server error"
            break

        if model is None:
            model = chunk.model
        
        # Collect stats if available (not standard in chunk yet but good to have)
        if chunk.stats:
            prompt_tokens = chunk.stats.prompt_tokens
            completion_tokens = chunk.stats.generation_tokens

        if isinstance(chunk, TokenChunk):
            text_parts.append(chunk.text)
            if chunk.logprob is not None:
                logprobs_content.append(
                    LogprobsContentItem(
                        token=chunk.text,
                        logprob=chunk.logprob,
                        top_logprobs=chunk.top_logprobs or [],
                    )
                )
            
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason

    if error_message is not None:
        raise ValueError(error_message)

    combined_text = "".join(text_parts)
    assert model is not None

    return CompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=model,
        choices=[
            CompletionChoice(
                text=combined_text,
                index=0,
                logprobs=Logprobs(content=logprobs_content)
                if logprobs_content
                else None,
                finish_reason=finish_reason,
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )
