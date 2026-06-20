"""Sandbox execution package for browser-use

This package provides type-safe sandbox code execution with SSE streaming.

Example:
    from browser_use.sandbox import sandbox, SSEEvent, SSEEventType

    @sandbox(log_level="INFO")
    async def my_task(browser: Browser) -> str:
        page = await browser.get_current_page()
        await page.goto("https://example.com")
        return await page.title()

    result = await my_task()
"""

from browser_use.sandbox.sandbox import (
	CLOUDPICKLE_AVAILABLE,
	SandboxError,
	_require_cloudpickle,
	sandbox,
)
from browser_use.sandbox.views import (
	BrowserCreatedData,
	ErrorData,
	ExecutionResponse,
	LogData,
	ResultData,
	SSEEvent,
	SSEEventType,
)

__all__ = [
	# Main decorator
	'sandbox',
	'SandboxError',
	# Optional dep helpers
	'CLOUDPICKLE_AVAILABLE',
	'_require_cloudpickle',
	# Event types
	'SSEEvent',
	'SSEEventType',
	# Event data models
	'BrowserCreatedData',
	'LogData',
	'ResultData',
	'ErrorData',
	'ExecutionResponse',
]
