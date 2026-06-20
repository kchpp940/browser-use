from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from browser_use.skills.service import SkillService
	from browser_use.skills.views import MissingCookieException

_LAZY_IMPORTS = {
	'SkillService': ('browser_use.skills.service', 'SkillService'),
	'MissingCookieException': ('browser_use.skills.views', 'MissingCookieException'),
}


def __getattr__(name: str):
	"""Lazy import — defer loading SkillService (and its browser-use-sdk dep) until accessed."""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		from importlib import import_module

		module = import_module(module_path)
		attr = getattr(module, attr_name)
		globals()[name] = attr
		return attr
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ['SkillService', 'MissingCookieException']
