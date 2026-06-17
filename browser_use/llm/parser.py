from __future__ import annotations

import json
import logging
import re
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger(__name__)


class NormalizedToolCall(BaseModel):
	"""Normalized tool call representation across all providers."""

	id: str | None = None
	name: str
	arguments: dict[str, Any] | str  # dict or raw JSON string


class NormalizedLLMResponse(BaseModel):
	"""
	Normalized intermediate response structure shared across all LLM providers.

	Every provider's chat.py should convert its native response format into this
	structure BEFORE passing it to AgentOutputParser. This unifies the parsing
	pipeline so the same validation/error-recovery logic applies to all models.
	"""

	# Raw text content extracted from the response (may contain JSON wrapped in markdown)
	raw_text: str = ''

	# Pre-parsed dict, if the provider already deserialized JSON (e.g. Anthropic tool_use.input, Gemini .parsed)
	parsed_dict: dict[str, Any] | None = None

	# Refusal / safety block content
	refusal: str | None = None

	# Structured tool/function calls
	tool_calls: list[NormalizedToolCall] = []

	# Thinking / reasoning content
	thinking: str | None = None
	redacted_thinking: str | None = None

	# Provider-specific metadata
	stop_reason: str | None = None
	stop_details: dict[str, Any] | None = None


class AgentOutputParser(Generic[T]):
	"""
	Unified parser that converts a NormalizedLLMResponse into a typed Pydantic model (e.g. AgentOutput).

	Implements multi-strategy JSON extraction and error recovery:
	  1. parsed_dict (pre-parsed by provider)
	  2. strict JSON from raw_text (model_validate_json)
	  3. multi candidate JSON extraction (markdown code blocks, brace matching)
	  4. dict validation fallback (model_validate on JSON.loads result)
	"""

	def __init__(self, output_format: type[T]):
		self.output_format = output_format

	# ------------------------------------------------------------------
	# JSON extraction utilities
	# ------------------------------------------------------------------
	@staticmethod
	def _strip_markdown_code(text: str) -> str:
		"""Strip ```json ... ``` or ``` ... ``` wrappers."""
		stripped = text.strip()
		pattern = re.compile(r'^```(?:json)?\s*\n?(.*?)\n?```$', re.DOTALL | re.IGNORECASE)
		match = pattern.match(stripped)
		if match:
			return match.group(1).strip()
		return stripped

	@staticmethod
	def _find_first_json(text: str) -> str | None:
		"""Find the first top-level { ... } or [ ... ] block in text."""
		stripped = text.strip()
		if not stripped:
			return None

		for opener, closer in (('{', '}'), ('[', ']')):
			start = stripped.find(opener)
			if start == -1:
				continue
			depth = 0
			in_string = False
			escape = False
			for i in range(start, len(stripped)):
				ch = stripped[i]
				if escape:
					escape = False
					continue
				if ch == '\\':
					escape = True
					continue
				if ch == '"':
					in_string = not in_string
					continue
				if in_string:
					continue
				if ch == opener:
					depth += 1
				elif ch == closer:
					depth -= 1
					if depth == 0:
						return stripped[start : i + 1]
		return None

	@classmethod
	def _generate_candidates(cls, text: str) -> list[str]:
		"""Generate multiple candidate JSON strings from raw model text."""
		if not text:
			return []
		candidates: list[str] = []
		stripped = text.strip()

		# 1) raw text (possibly after whitespace strip)
		if stripped:
			candidates.append(stripped)

		# 2) markdown code block stripped
		without_md = cls._strip_markdown_code(stripped)
		if without_md and without_md != stripped:
			candidates.append(without_md)

		# 3) brace / bracket matched region
		json_block = cls._find_first_json(stripped)
		if json_block and json_block != without_md and json_block != stripped:
			candidates.append(json_block)

		# 4) try extracting code-block content even inside larger text
		code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', stripped, re.DOTALL | re.IGNORECASE)
		if code_block_match:
			inner = code_block_match.group(1).strip()
			if inner and inner not in candidates:
				candidates.append(inner)

		# Deduplicate preserving order
		seen: set[str] = set()
		result: list[str] = []
		for c in candidates:
			if c not in seen:
				seen.add(c)
				result.append(c)
		return result

	# ------------------------------------------------------------------
	# Validation helpers
	# ------------------------------------------------------------------
	def _try_validate_json(self, candidate: str) -> T | None:
		"""Try strict model_validate_json."""
		try:
			return self.output_format.model_validate_json(candidate)
		except (ValidationError, json.JSONDecodeError):
			return None

	def _try_validate_dict(self, data: dict[str, Any]) -> T | None:
		"""Try model_validate on a pre-parsed dict."""
		try:
			return self.output_format.model_validate(data)
		except ValidationError as e:
			logger.debug(f'Dict validation failed: {e}')
			return None

	# ------------------------------------------------------------------
	# Tool-call extraction (for Anthropic / tool-calling paths)
	# ------------------------------------------------------------------
	def _extract_tool_arguments(self, normalized: NormalizedLLMResponse) -> dict[str, Any] | None:
		"""Extract the first tool call's arguments as a dict."""
		if not normalized.tool_calls:
			return None
		tc = normalized.tool_calls[0]
		args = tc.arguments
		if isinstance(args, dict):
			return args
		if isinstance(args, str):
			try:
				return json.loads(args)
			except json.JSONDecodeError:
				return None
		return None

	# ------------------------------------------------------------------
	# Main entry point
	# ------------------------------------------------------------------
	def parse(self, normalized: NormalizedLLMResponse) -> T:
		"""
		Parse a NormalizedLLMResponse into the target Pydantic model.

		Raises ModelParseError if every strategy fails.
		"""
		output_name = self.output_format.__name__

		# Strategy 1: provider already gave us a parsed dict
		if normalized.parsed_dict is not None:
			result = self._try_validate_dict(normalized.parsed_dict)
			if result is not None:
				return result
			logger.debug(f'Strategy 1 (parsed_dict) failed for {output_name}')

		# Strategy 2: provider gave us a structured tool-call (Anthropic / Groq tool-calling path)
		tool_args = self._extract_tool_arguments(normalized)
		if tool_args is not None:
			result = self._try_validate_dict(tool_args)
			if result is not None:
				return result
			logger.debug(f'Strategy 2 (tool_call args) failed for {output_name}')

			# also try converting tool_args back to JSON for strict parsing
			try:
				tool_args_json = json.dumps(tool_args)
				result = self._try_validate_json(tool_args_json)
				if result is not None:
					return result
			except (TypeError, ValueError):
				pass

		# Strategy 3: multi-candidate JSON extraction from raw_text
		candidates = self._generate_candidates(normalized.raw_text)
		if candidates:
			# 3a: strict JSON validation first
			for cand in candidates:
				result = self._try_validate_json(cand)
				if result is not None:
					return result
			logger.debug(f'Strategy 3a (model_validate_json) failed for {output_name}, trying dict validation')

			# 3b: then dict-based validation (more forgiving)
			for cand in candidates:
				try:
					data = json.loads(cand)
				except json.JSONDecodeError:
					continue
				if isinstance(data, dict):
					result = self._try_validate_dict(data)
					if result is not None:
						return result

		# All strategies failed — report with diagnostic info
		info_parts = []
		if normalized.parsed_dict is not None:
			info_parts.append(f'parsed_dict keys={list(normalized.parsed_dict.keys())}')
		if normalized.tool_calls:
			info_parts.append(f'tool_calls={[(tc.name, type(tc.arguments).__name__) for tc in normalized.tool_calls]}')
		if normalized.refusal:
			info_parts.append(f'refusal={normalized.refusal[:100]!r}')
		info_parts.append(f'raw_text[:300]={normalized.raw_text[:300]!r}')
		diagnostic = ' | '.join(info_parts)

		from browser_use.llm.exceptions import ModelParseError

		raise ModelParseError(
			message=f'Failed to parse LLM response into {output_name}. {diagnostic}',
			model='unknown',
		)
