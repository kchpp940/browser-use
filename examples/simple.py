"""
Setup:
1. Get your API key from https://cloud.browser-use.com/new-api-key
2. Set environment variable: export BROWSER_USE_API_KEY="your-key"
"""

from dotenv import load_dotenv

from browser_use import Agent, ChatBrowserUse, ResultSerializer

load_dotenv()


async def main():
	agent = Agent(
		task='Find the number of stars of the following repos: browser-use, playwright, stagehand, react, nextjs',
		llm=ChatBrowserUse(model='bu-2-0'),
	)

	# Use run_with_result() to obtain a structured RuntimeExecutionResult.
	# You can then pick a format via ResultSerializer:
	#   serializer.to_text()        — human-readable summary (CLI default)
	#   serializer.to_json()        — compact JSON for piping
	#   serializer.to_json(indent=2) — pretty JSON for debugging
	#   serializer.output(mode='both') — human text + JSON block (MCP-style)
	result = await agent.run_with_result()

	serializer = ResultSerializer(result)
	# Default: pretty human-readable summary.
	# Uncomment one of the following for structured output:
	# print(serializer.to_json(indent=2))
	# print(serializer.output(mode='both'))
	print(serializer.to_text())


if __name__ == '__main__':
	import asyncio

	asyncio.run(main())
