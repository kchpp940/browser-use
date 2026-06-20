from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from browser_use.llm.google.chat import ChatGoogle

_LAZY_IMPORTS = {
	'ChatGoogle': ('browser_use.llm.google.chat', 'ChatGoogle'),
}


def __getattr__(name: str):
	"""Lazy import — defer loading ChatGoogle (and google-genai SDK) until actually accessed."""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		from importlib import import_module

		module = import_module(module_path)
		attr = getattr(module, attr_name)
		globals()[name] = attr
		return attr
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ['ChatGoogle']
