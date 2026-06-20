"""Utility functions for converting between RuntimeConfig and other config formats.

This module provides bridges between the unified RuntimeConfig and existing
configuration objects like BrowserProfile.
"""

from __future__ import annotations

from typing import Any

from browser_use.browser.profile import BrowserProfile, ProxySettings, ViewportSize
from browser_use.runtime_config.models import BrowserConfig, RuntimeConfig


def browser_config_to_profile_dict(config: BrowserConfig) -> dict[str, Any]:
	"""Convert BrowserConfig to a dict compatible with BrowserProfile.

	Args:
	    config: BrowserConfig instance

	Returns:
	    Dict that can be passed to BrowserProfile(**result)
	"""
	result: dict[str, Any] = {}

	# Basic browser settings
	if config.headless is not None:
		result['headless'] = config.headless

	if config.window_width is not None or config.window_height is not None:
		width = config.window_width or 1920
		height = config.window_height or 1080
		result['window_size'] = ViewportSize(width=width, height=height)

	if config.user_data_dir is not None:
		result['user_data_dir'] = config.user_data_dir

	if config.profile_directory:
		result['profile_directory'] = config.profile_directory

	if config.cdp_url is not None:
		result['cdp_url'] = config.cdp_url

	if config.channel is not None:
		result['channel'] = config.channel

	if config.executable_path is not None:
		result['executable_path'] = config.executable_path

	# Cloud browser settings
	result['use_cloud'] = config.use_cloud

	if config.cloud_profile_id is not None:
		result['cloud_profile_id'] = config.cloud_profile_id

	if config.cloud_proxy_country_code is not None:
		result['cloud_proxy_country_code'] = config.cloud_proxy_country_code

	if config.cloud_timeout is not None:
		result['cloud_timeout'] = config.cloud_timeout

	# Domain settings
	if config.allowed_domains is not None:
		result['allowed_domains'] = config.allowed_domains

	if config.prohibited_domains is not None:
		result['prohibited_domains'] = config.prohibited_domains

	result['block_ip_addresses'] = config.block_ip_addresses

	# Proxy settings
	if config.proxy_server is not None:
		proxy = ProxySettings(
			server=config.proxy_server,
			bypass=config.proxy_bypass,
			username=config.proxy_username,
			password=config.proxy_password,
		)
		result['proxy'] = proxy

	# Feature settings
	result['enable_default_extensions'] = config.enable_default_extensions

	if config.keep_alive is not None:
		result['keep_alive'] = config.keep_alive

	if config.downloads_path is not None:
		result['downloads_path'] = config.downloads_path

	# Timing settings
	result['minimum_wait_page_load_time'] = config.minimum_wait_page_load_time
	result['wait_for_network_idle_page_load_time'] = config.wait_for_network_idle_page_load_time
	result['wait_between_actions'] = config.wait_between_actions

	# UI/viewport settings
	result['highlight_elements'] = config.highlight_elements
	result['paint_order_filtering'] = config.paint_order_filtering

	# Recording settings
	if config.record_video_dir is not None:
		result['record_video_dir'] = config.record_video_dir

	if config.record_har_path is not None:
		result['record_har_path'] = config.record_har_path

	if config.traces_dir is not None:
		result['traces_dir'] = config.traces_dir

	# Iframe settings
	result['cross_origin_iframes'] = config.cross_origin_iframes

	# Security settings
	result['disable_security'] = config.disable_security
	result['deterministic_rendering'] = config.deterministic_rendering

	# DevTools
	result['devtools'] = config.devtools

	if config.chromium_sandbox is not None:
		result['chromium_sandbox'] = config.chromium_sandbox

	# Downloads
	result['auto_download_pdfs'] = config.auto_download_pdfs

	# Demo mode
	result['demo_mode'] = config.demo_mode

	# Local browser flag
	result['is_local'] = config.is_local

	# Permissions
	if config.permissions:
		result['permissions'] = config.permissions

	return result


def create_browser_profile_from_config(config: RuntimeConfig | BrowserConfig) -> BrowserProfile:
	"""Create a BrowserProfile from RuntimeConfig or BrowserConfig.

	Args:
	    config: RuntimeConfig or BrowserConfig instance

	Returns:
	    BrowserProfile instance with settings from the config
	"""
	if isinstance(config, RuntimeConfig):
		browser_config = config.browser
	else:
		browser_config = config

	profile_dict = browser_config_to_profile_dict(browser_config)
	return BrowserProfile(**profile_dict)


def update_browser_profile_from_config(
	profile: BrowserProfile,
	config: RuntimeConfig | BrowserConfig,
) -> BrowserProfile:
	"""Update an existing BrowserProfile with settings from config.

	Only non-default values from config are applied.

	Args:
	    profile: Existing BrowserProfile to update
	    config: RuntimeConfig or BrowserConfig with new settings

	Returns:
	    Updated BrowserProfile (new instance)
	"""
	if isinstance(config, RuntimeConfig):
		browser_config = config.browser
	else:
		browser_config = config

	update_dict = browser_config_to_profile_dict(browser_config)
	return profile.model_copy(update=update_dict)
