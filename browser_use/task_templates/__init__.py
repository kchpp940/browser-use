"""Task templates for reusable browser automation workflows.

This module provides a template system that allows users to define reusable
browser automation tasks with variables, default configurations, and
structured output.

Example:
    >>> from browser_use.task_templates import TaskTemplate, TemplateManager
    >>> # Create a template
    >>> template = TaskTemplate(
    ...     name="hn_top_post",
    ...     description="Find the top post on Hacker News",
    ...     prompt_template="Go to Hacker News and find the {{ position }} post. Extract its title, URL, and points.",
    ...     variables={
    ...         "position": TemplateVariable(type="string", default="number 1", description="Position to find (e.g., 'number 1', 'top 3')")
    ...     },
    ...     default_tools=["search", "navigate", "click", "extract", "done"],
    ... )
    >>> # Save and run
    >>> manager = TemplateManager()
    >>> manager.save(template)
    >>> result = await manager.run("hn_top_post", variables={"position": "top 5"})
"""

from browser_use.task_templates.views import (
	OutputFileRule,
	TaskTemplate,
	TaskTemplateExecutionResult,
	TaskTemplateStatus,
	TemplateVariable,
	TemplateVariableType,
)
from browser_use.task_templates.service import TemplateManager, get_template_manager, run_template

__all__ = [
	'TaskTemplate',
	'TemplateVariable',
	'TemplateVariableType',
	'OutputFileRule',
	'TaskTemplateExecutionResult',
	'TaskTemplateStatus',
	'TemplateManager',
	'get_template_manager',
	'run_template',
]
