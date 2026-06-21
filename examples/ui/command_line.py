"""
To Use It:

Example 1: Using OpenAI (default), with default task: 'go to reddit and search for posts about browser-use'
python command_line.py

Example 2: Using OpenAI with a Custom Query
python command_line.py --query "go to google and search for browser-use"

Example 3: Using Anthropic's Claude Model with a Custom Query
python command_line.py --query "find latest Python tutorials on Medium" --provider anthropic

"""

import argparse
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from browser_use.runtime import BrowserSessionConfig, CommandRuntimeAdapter, LLMConfig, TaskConfig


def parse_arguments():
	parser = argparse.ArgumentParser(description='Automate browser tasks using an LLM agent.')
	parser.add_argument(
		'--query', type=str, help='The query to process', default='go to reddit and search for posts about browser-use'
	)
	parser.add_argument(
		'--provider',
		type=str,
		choices=['openai', 'anthropic'],
		default='openai',
		help='The model provider to use (default: openai)',
	)
	return parser.parse_args()


def build_task_config(query: str, provider: str) -> TaskConfig:
	llm_config = LLMConfig(provider=provider)
	if provider == 'anthropic':
		api_key = os.getenv('ANTHROPIC_API_KEY')
		if not api_key:
			raise ValueError('Error: ANTHROPIC_API_KEY is not set. Please provide a valid API key.')
		llm_config.model = 'claude-3-5-sonnet-20240620'
		llm_config.temperature = 0.0
	elif provider == 'openai':
		api_key = os.getenv('OPENAI_API_KEY')
		if not api_key:
			raise ValueError('Error: OPENAI_API_KEY is not set. Please provide a valid API key.')
		llm_config.model = 'gpt-4.1'
		llm_config.temperature = 0.0

	return TaskConfig(
		task=query,
		llm=llm_config,
		browser=BrowserSessionConfig(),
		use_vision=True,
		agent_settings={'max_actions_per_step': 1},
	)


async def main():
	args = parse_arguments()
	task_config = build_task_config(args.query, args.provider)
	adapter = CommandRuntimeAdapter(task_config)
	result = await adapter.run_agent(max_steps=25)
	print(result.format_text())


if __name__ == '__main__':
	asyncio.run(main())
