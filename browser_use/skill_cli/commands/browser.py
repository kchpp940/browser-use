"""Browser control commands."""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any

from browser_use.agent.views import ToolExecutionResult
from browser_use.skill_cli.sessions import SessionInfo

logger = logging.getLogger(__name__)

COMMANDS = {
	'open',
	'click',
	'type',
	'input',
	'scroll',
	'back',
	'screenshot',
	'state',
	'tab',
	'keys',
	'select',
	'upload',
	'eval',
	'extract',
	'cookies',
	'wait',
	'hover',
	'dblclick',
	'rightclick',
	'get',
	'record',
}


def _wrap_result(action: str, result: dict[str, Any]) -> dict[str, Any]:
	"""Wrap a command result dict with a ToolExecutionResult under ``_structured``."""
	error = result.get('error')
	ter = ToolExecutionResult(
		tool_name=action,
		success=error is None,
		message=str(error) if error else '',
		data={k: v for k, v in result.items() if k != 'error'},
		error=str(error) if error else None,
		url=result.get('url'),
	)
	result['_structured'] = ter.model_dump(exclude={'data'})
	return result


async def _execute_js(session: SessionInfo, js: str) -> Any:
	"""Execute JavaScript in the browser via CDP."""
	bs = session.browser_session
	# Get or create a CDP session for the focused target
	cdp_session = await bs.get_or_create_cdp_session(target_id=None, focus=False)
	if not cdp_session:
		raise RuntimeError('No active browser session')

	result = await cdp_session.cdp_client.send.Runtime.evaluate(
		params={'expression': js, 'returnByValue': True},
		session_id=cdp_session.session_id,
	)
	return result.get('result', {}).get('value')


async def _get_element_center(session: SessionInfo, node: Any) -> tuple[float, float] | None:
	"""Get the center coordinates of an element."""
	bs = session.browser_session
	try:
		cdp_session = await bs.cdp_client_for_node(node)
		session_id = cdp_session.session_id
		backend_node_id = node.backend_node_id

		# Scroll element into view first
		try:
			await cdp_session.cdp_client.send.DOM.scrollIntoViewIfNeeded(
				params={'backendNodeId': backend_node_id}, session_id=session_id
			)
			await asyncio.sleep(0.05)
		except Exception:
			pass

		# Get element coordinates
		element_rect = await bs.get_element_coordinates(backend_node_id, cdp_session)
		if element_rect:
			center_x = element_rect.x + element_rect.width / 2
			center_y = element_rect.y + element_rect.height / 2
			return center_x, center_y
		return None
	except Exception as e:
		logger.error(f'Failed to get element center: {e}')
		return None


async def handle(action: str, session: SessionInfo, params: dict[str, Any]) -> Any:
	"""Handle browser control command."""
	bs = session.browser_session
	actions = session.actions
	if actions is None:
		return _wrap_result(action, {'error': 'ActionHandler not initialized'})

	if action == 'open':
		url = params['url']
		if not url.startswith(('http://', 'https://', 'file://')):
			url = 'https://' + url
		await actions.navigate(url)
		result: dict[str, Any] = {'url': url}
		if bs.browser_profile.use_cloud and bs.cdp_url:
			from urllib.parse import quote

			result['live_url'] = f'https://live.browser-use.com/?wss={quote(bs.cdp_url, safe="")}'
		return _wrap_result(action, result)

	elif action == 'click':
		args = params.get('args', [])
		if len(args) == 2:
			x, y = args
			await actions.click_coordinate(x, y)
			return _wrap_result(action, {'clicked_coordinate': {'x': x, 'y': y}})
		elif len(args) == 1:
			index = args[0]
			node = await bs.get_element_by_index(index)
			if node is None:
				return _wrap_result(action, {'error': f'Element index {index} not found - page may have changed'})
			await actions.click_element(node)
			return _wrap_result(action, {'clicked': index})
		else:
			return _wrap_result(action, {'error': 'Usage: click <index> or click <x> <y>'})

	elif action == 'type':
		text = params['text']
		cdp_session = await bs.get_or_create_cdp_session(target_id=None, focus=False)
		if not cdp_session:
			return _wrap_result(action, {'error': 'No active browser session'})
		await cdp_session.cdp_client.send.Input.insertText(
			params={'text': text},
			session_id=cdp_session.session_id,
		)
		return _wrap_result(action, {'typed': text})

	elif action == 'input':
		index = params['index']
		text = params['text']
		node = await bs.get_element_by_index(index)
		if node is None:
			return _wrap_result(action, {'error': f'Element index {index} not found - page may have changed'})
		await actions.click_element(node)
		await actions.type_text(node, text)
		return _wrap_result(action, {'input': text, 'element': index})

	elif action == 'scroll':
		direction = params.get('direction', 'down')
		amount = params.get('amount', 500)
		await actions.scroll(direction, amount)
		return _wrap_result(action, {'scrolled': direction, 'amount': amount})

	elif action == 'back':
		await actions.go_back()
		return _wrap_result(action, {'back': True})

	elif action == 'screenshot':
		data = await bs.take_screenshot(full_page=params.get('full', False))

		if params.get('path'):
			path = Path(params['path'])
			path.write_bytes(data)
			return _wrap_result(action, {'saved': str(path), 'size': len(data)})

		return _wrap_result(action, {'screenshot': base64.b64encode(data).decode(), 'size': len(data)})

	elif action == 'state':
		state = await actions.get_state()
		assert state.dom_state is not None
		state_text = state.dom_state.llm_representation()

		# Prepend viewport dimensions
		if state.page_info:
			pi = state.page_info
			viewport_text = f'viewport: {pi.viewport_width}x{pi.viewport_height}\n'
			viewport_text += f'page: {pi.page_width}x{pi.page_height}\n'
			viewport_text += f'scroll: ({pi.scroll_x}, {pi.scroll_y})\n'
			state_text = viewport_text + state_text

		# Append auto-dismissed popup messages
		if bs._closed_popup_messages:
			state_text += '\nAuto-closed dialogs:\n'
			for msg in bs._closed_popup_messages:
				state_text += f'  {msg}\n'
			bs._closed_popup_messages.clear()

		return _wrap_result(action, {'_raw_text': state_text})

	elif action == 'tab':
		tab_command = params.get('tab_command')

		if tab_command == 'list':
			page_targets = bs.session_manager.get_all_page_targets() if bs.session_manager else []
			lines = ['TAB  URL']
			for i, t in enumerate(page_targets):
				lines.append(f'{i:<4} {t.url}')
			return _wrap_result(action, {'_raw_text': '\n'.join(lines)})

		elif tab_command == 'new':
			url = params.get('url', 'about:blank')
			target_id = await bs._cdp_create_new_page(url, background=True)
			bs.agent_focus_target_id = target_id
			return _wrap_result(action, {'created': target_id[:8], 'url': url})

		elif tab_command == 'switch':
			tab_index = params['tab']
			page_targets = bs.session_manager.get_all_page_targets() if bs.session_manager else []
			if tab_index < 0 or tab_index >= len(page_targets):
				return _wrap_result(action, {'error': f'Invalid tab index {tab_index}. Available: 0-{len(page_targets) - 1}'})
			bs.agent_focus_target_id = page_targets[tab_index].target_id
			return _wrap_result(action, {'switched': tab_index})

		elif tab_command == 'close':
			tab_indices = params.get('tabs', [])

			page_targets = bs.session_manager.get_all_page_targets() if bs.session_manager else []

			async def _close_target(tid: str) -> None:
				cdp_session = await bs.get_or_create_cdp_session(target_id=None, focus=False)
				if cdp_session:
					await cdp_session.cdp_client.send.Target.closeTarget(params={'targetId': tid})

			if not tab_indices:
				# Use caller's logical focus, not Chrome's global focus
				target_id = bs.agent_focus_target_id
				if not target_id:
					target_id = bs.session_manager.get_focused_target().target_id if bs.session_manager else None
				if not target_id:
					return _wrap_result(action, {'error': 'No focused tab to close'})
				await _close_target(target_id)
				return _wrap_result(action, {'closed': [0]})

			closed = []
			errors = []
			for idx in sorted(tab_indices, reverse=True):
				if idx < 0 or idx >= len(page_targets):
					errors.append(f'Tab {idx} out of range')
					continue
				try:
					await _close_target(page_targets[idx].target_id)
					closed.append(idx)
				except Exception as e:
					errors.append(f'Tab {idx}: {e}')
			result: dict[str, Any] = {'closed': closed}
			if errors:
				result['errors'] = errors
			return _wrap_result(action, result)

		return _wrap_result(action, {'error': 'Invalid tab command. Use: list, new, switch, close'})

	elif action == 'keys':
		keys = params['keys']
		await actions.send_keys(keys)
		return _wrap_result(action, {'sent': keys})

	elif action == 'select':
		index = params['index']
		value = params['value']
		node = await bs.get_element_by_index(index)
		if node is None:
			return _wrap_result(action, {'error': f'Element index {index} not found - page may have changed'})
		await actions.select_dropdown(node, value)
		return _wrap_result(action, {'selected': value, 'element': index})

	elif action == 'upload':
		index = params['index']
		file_path = params['path']

		p = Path(file_path)
		if not p.exists():
			return _wrap_result(action, {'error': f'File not found: {file_path}'})
		if not p.is_file():
			return _wrap_result(action, {'error': f'Not a file: {file_path}'})
		if p.stat().st_size == 0:
			return _wrap_result(action, {'error': f'File is empty (0 bytes): {file_path}'})

		node = await bs.get_element_by_index(index)
		if node is None:
			return _wrap_result(action, {'error': f'Element index {index} not found - page may have changed'})

		file_input_node = bs.find_file_input_near_element(node)

		if file_input_node is None:
			selector_map = await bs.get_selector_map()
			file_input_indices = [idx for idx, el in selector_map.items() if bs.is_file_input(el)]
			if file_input_indices:
				hint = f' File input(s) found at index: {", ".join(map(str, file_input_indices))}'
			else:
				hint = ' No file input found on the page.'
			return _wrap_result(action, {'error': f'Element {index} is not a file input.{hint}'})

		await actions.upload_file(file_input_node, file_path)
		return _wrap_result(action, {'uploaded': file_path, 'element': index})

	elif action == 'eval':
		js = params['js']
		result = await _execute_js(session, js)
		return _wrap_result(action, {'result': result})

	elif action == 'extract':
		query = params['query']
		return _wrap_result(action, {'query': query, 'error': 'extract is not yet implemented'})

	elif action == 'hover':
		index = params['index']
		node = await bs.get_element_by_index(index)
		if node is None:
			return _wrap_result(action, {'error': f'Element index {index} not found - page may have changed'})

		coords = await _get_element_center(session, node)
		if not coords:
			return _wrap_result(action, {'error': 'Could not get element coordinates for hover'})

		center_x, center_y = coords
		cdp_session = await bs.cdp_client_for_node(node)
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mouseMoved', 'x': center_x, 'y': center_y},
			session_id=cdp_session.session_id,
		)
		return _wrap_result(action, {'hovered': index})

	elif action == 'dblclick':
		index = params['index']
		node = await bs.get_element_by_index(index)
		if node is None:
			return _wrap_result(action, {'error': f'Element index {index} not found - page may have changed'})

		coords = await _get_element_center(session, node)
		if not coords:
			return _wrap_result(action, {'error': 'Could not get element coordinates for double-click'})

		center_x, center_y = coords
		cdp_session = await bs.cdp_client_for_node(node)
		session_id = cdp_session.session_id

		# Move mouse to element
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mouseMoved', 'x': center_x, 'y': center_y},
			session_id=session_id,
		)
		await asyncio.sleep(0.05)

		# Double click (clickCount: 2)
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={
				'type': 'mousePressed',
				'x': center_x,
				'y': center_y,
				'button': 'left',
				'clickCount': 2,
			},
			session_id=session_id,
		)
		await asyncio.sleep(0.05)

		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={
				'type': 'mouseReleased',
				'x': center_x,
				'y': center_y,
				'button': 'left',
				'clickCount': 2,
			},
			session_id=session_id,
		)
		return _wrap_result(action, {'double_clicked': index})

	elif action == 'rightclick':
		index = params['index']
		node = await bs.get_element_by_index(index)
		if node is None:
			return _wrap_result(action, {'error': f'Element index {index} not found - page may have changed'})

		coords = await _get_element_center(session, node)
		if not coords:
			return _wrap_result(action, {'error': 'Could not get element coordinates for right-click'})

		center_x, center_y = coords
		cdp_session = await bs.cdp_client_for_node(node)
		session_id = cdp_session.session_id

		# Move mouse to element
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={'type': 'mouseMoved', 'x': center_x, 'y': center_y},
			session_id=session_id,
		)
		await asyncio.sleep(0.05)

		# Right click (button: 'right')
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={
				'type': 'mousePressed',
				'x': center_x,
				'y': center_y,
				'button': 'right',
				'clickCount': 1,
			},
			session_id=session_id,
		)
		await asyncio.sleep(0.05)

		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={
				'type': 'mouseReleased',
				'x': center_x,
				'y': center_y,
				'button': 'right',
				'clickCount': 1,
			},
			session_id=session_id,
		)
		return _wrap_result(action, {'right_clicked': index})

	elif action == 'cookies':
		cookies_command = params.get('cookies_command')

		if cookies_command == 'get':
			# Get cookies via direct CDP
			cookies = await bs._cdp_get_cookies()
			# Convert Cookie objects to dicts
			cookie_list: list[dict[str, Any]] = []
			for c in cookies:
				cookie_dict: dict[str, Any] = {
					'name': c.get('name', ''),
					'value': c.get('value', ''),
					'domain': c.get('domain', ''),
					'path': c.get('path', '/'),
					'secure': c.get('secure', False),
					'httpOnly': c.get('httpOnly', False),
				}
				if 'sameSite' in c:
					cookie_dict['sameSite'] = c.get('sameSite')
				if 'expires' in c:
					cookie_dict['expires'] = c.get('expires')
				cookie_list.append(cookie_dict)

			# Filter by URL if provided
			url = params.get('url')
			if url:
				from urllib.parse import urlparse

				parsed = urlparse(url)
				domain = parsed.netloc
				cookie_list = [
					c
					for c in cookie_list
					if domain.endswith(str(c.get('domain', '')).lstrip('.'))
					or str(c.get('domain', '')).lstrip('.').endswith(domain)
				]

			return _wrap_result(action, {'cookies': cookie_list})

		elif cookies_command == 'set':
			from cdp_use.cdp.network import Cookie

			cookie_dict: dict[str, Any] = {
				'name': params['name'],
				'value': params['value'],
				'path': params.get('path', '/'),
				'secure': params.get('secure', False),
				'httpOnly': params.get('http_only', False),
			}

			if params.get('domain'):
				cookie_dict['domain'] = params['domain']
			if params.get('same_site'):
				cookie_dict['sameSite'] = params['same_site']
			if params.get('expires'):
				cookie_dict['expires'] = params['expires']

			# If no domain specified, get current URL's domain
			if not params.get('domain'):
				hostname = await _execute_js(session, 'window.location.hostname')
				if hostname:
					cookie_dict['domain'] = hostname

			try:
				cookie_obj = Cookie(**cookie_dict)
				await bs._cdp_set_cookies([cookie_obj])
				return _wrap_result(action, {'set': params['name'], 'success': True})
			except Exception as e:
				logger.error(f'Failed to set cookie: {e}')
				return _wrap_result(action, {'set': params['name'], 'success': False, 'error': str(e)})

		elif cookies_command == 'clear':
			url = params.get('url')
			if url:
				# Clear cookies only for specific URL domain
				from urllib.parse import urlparse

				cookies = await bs._cdp_get_cookies()
				parsed = urlparse(url)
				domain = parsed.netloc

				cdp_session = await bs.get_or_create_cdp_session(target_id=None, focus=False)
				if cdp_session:
					for cookie in cookies:
						cookie_domain = str(cookie.get('domain', '')).lstrip('.')
						if domain.endswith(cookie_domain) or cookie_domain.endswith(domain):
							await cdp_session.cdp_client.send.Network.deleteCookies(
								params={
									'name': cookie.get('name', ''),
									'domain': cookie.get('domain'),
									'path': cookie.get('path', '/'),
								},
								session_id=cdp_session.session_id,
							)
			else:
				# Clear all cookies
				await bs._cdp_clear_cookies()

			return _wrap_result(action, {'cleared': True, 'url': url})

		elif cookies_command == 'export':
			import json

			# Get cookies via direct CDP
			cookies = await bs._cdp_get_cookies()
			# Convert to list of dicts
			cookie_list: list[dict[str, Any]] = []
			for c in cookies:
				cookie_dict: dict[str, Any] = {
					'name': c.get('name', ''),
					'value': c.get('value', ''),
					'domain': c.get('domain', ''),
					'path': c.get('path', '/'),
					'secure': c.get('secure', False),
					'httpOnly': c.get('httpOnly', False),
				}
				if 'sameSite' in c:
					cookie_dict['sameSite'] = c.get('sameSite')
				if 'expires' in c:
					cookie_dict['expires'] = c.get('expires')
				cookie_list.append(cookie_dict)

			# Filter by URL if provided
			url = params.get('url')
			if url:
				from urllib.parse import urlparse

				parsed = urlparse(url)
				domain = parsed.netloc
				cookie_list = [
					c
					for c in cookie_list
					if domain.endswith(str(c.get('domain', '')).lstrip('.'))
					or str(c.get('domain', '')).lstrip('.').endswith(domain)
				]

			file_path = Path(params['file'])
			file_path.write_text(json.dumps(cookie_list, indent=2, ensure_ascii=False), encoding='utf-8')
			return _wrap_result(action, {'exported': len(cookie_list), 'file': str(file_path)})

		elif cookies_command == 'import':
			import json

			file_path = Path(params['file'])
			if not file_path.exists():
				return _wrap_result(action, {'error': f'File not found: {file_path}'})

			cookies = json.loads(file_path.read_text())

			# Get CDP session for bulk cookie setting
			cdp_session = await bs.get_or_create_cdp_session(target_id=None, focus=False)
			if not cdp_session:
				return _wrap_result(action, {'error': 'No active browser session'})

			# Build cookie list for bulk set
			cookie_list = []
			for c in cookies:
				cookie_params = {
					'name': c['name'],
					'value': c['value'],
					'domain': c.get('domain'),
					'path': c.get('path', '/'),
					'secure': c.get('secure', False),
					'httpOnly': c.get('httpOnly', False),
				}
				if c.get('sameSite'):
					cookie_params['sameSite'] = c['sameSite']
				if c.get('expires'):
					cookie_params['expires'] = c['expires']
				cookie_list.append(cookie_params)

			# Set all cookies in one call
			try:
				await cdp_session.cdp_client.send.Network.setCookies(
					params={'cookies': cookie_list},  # type: ignore[arg-type]
					session_id=cdp_session.session_id,
				)
				return _wrap_result(action, {'imported': len(cookie_list), 'file': str(file_path)})
			except Exception as e:
				return _wrap_result(action, {'error': f'Failed to import cookies: {e}'})

		return _wrap_result(action, {'error': 'Invalid cookies command. Use: get, set, clear, export, import'})

	elif action == 'wait':
		import json as json_module

		wait_command = params.get('wait_command')

		if wait_command == 'selector':
			timeout_seconds = params.get('timeout', 30000) / 1000.0
			state = params.get('state', 'visible')
			selector = params['selector']
			poll_interval = 0.1
			elapsed = 0.0

			while elapsed < timeout_seconds:
				# Build JS check based on state
				if state == 'attached':
					js = f'document.querySelector({json_module.dumps(selector)}) !== null'
				elif state == 'detached':
					js = f'document.querySelector({json_module.dumps(selector)}) === null'
				elif state == 'visible':
					js = f"""
						(function() {{
							const el = document.querySelector({json_module.dumps(selector)});
							if (!el) return false;
							const style = window.getComputedStyle(el);
							const rect = el.getBoundingClientRect();
							return style.display !== 'none' &&
								   style.visibility !== 'hidden' &&
								   style.opacity !== '0' &&
								   rect.width > 0 &&
								   rect.height > 0;
						}})()
					"""
				elif state == 'hidden':
					js = f"""
						(function() {{
							const el = document.querySelector({json_module.dumps(selector)});
							if (!el) return true;
							const style = window.getComputedStyle(el);
							const rect = el.getBoundingClientRect();
							return style.display === 'none' ||
								   style.visibility === 'hidden' ||
								   style.opacity === '0' ||
								   rect.width === 0 ||
								   rect.height === 0;
						}})()
					"""
				else:
					js = f'document.querySelector({json_module.dumps(selector)}) !== null'

				result = await _execute_js(session, js)
				if result:
					return _wrap_result(action, {'selector': selector, 'found': True})

				await asyncio.sleep(poll_interval)
				elapsed += poll_interval

			return _wrap_result(action, {'selector': selector, 'found': False})

		elif wait_command == 'text':
			import json as json_module

			timeout_seconds = params.get('timeout', 30000) / 1000.0
			text = params['text']
			poll_interval = 0.1
			elapsed = 0.0

			while elapsed < timeout_seconds:
				js = f"""
					(function() {{
						const text = {json_module.dumps(text)};
						return document.body.innerText.includes(text);
					}})()
				"""
				result = await _execute_js(session, js)
				if result:
					return _wrap_result(action, {'text': text, 'found': True})

				await asyncio.sleep(poll_interval)
				elapsed += poll_interval

			return _wrap_result(action, {'text': text, 'found': False})

		return _wrap_result(action, {'error': 'Invalid wait command. Use: selector, text'})

	elif action == 'get':
		import json as json_module

		get_command = params.get('get_command')

		if get_command == 'title':
			title = await _execute_js(session, 'document.title')
			return _wrap_result(action, {'title': title or ''})

		elif get_command == 'html':
			selector = params.get('selector')
			if selector:
				js = f'(function(){{ const el = document.querySelector({json_module.dumps(selector)}); return el ? el.outerHTML : null; }})()'
			else:
				js = 'document.documentElement.outerHTML'
			html = await _execute_js(session, js)
			return _wrap_result(action, {'html': html or ''})

		elif get_command == 'text':
			index = params['index']
			node = await bs.get_element_by_index(index)
			if node is None:
				return _wrap_result(action, {'error': f'Element index {index} not found - page may have changed'})
			text = node.get_all_children_text(max_depth=10) if node else ''
			return _wrap_result(action, {'index': index, 'text': text})

		elif get_command == 'value':
			index = params['index']
			node = await bs.get_element_by_index(index)
			if node is None:
				return _wrap_result(action, {'error': f'Element index {index} not found - page may have changed'})

			try:
				cdp_session = await bs.cdp_client_for_node(node)
				resolve_result = await cdp_session.cdp_client.send.DOM.resolveNode(
					params={'backendNodeId': node.backend_node_id},
					session_id=cdp_session.session_id,
				)
				object_id = resolve_result['object'].get('objectId')  # type: ignore[union-attr]

				if object_id:
					value_result = await cdp_session.cdp_client.send.Runtime.callFunctionOn(
						params={
							'objectId': object_id,
							'functionDeclaration': 'function() { return this.value; }',
							'returnByValue': True,
						},
						session_id=cdp_session.session_id,
					)
					value = value_result.get('result', {}).get('value')
					return _wrap_result(action, {'index': index, 'value': value or ''})
				else:
					return _wrap_result(action, {'index': index, 'value': ''})
			except Exception as e:
				logger.error(f'Failed to get element value: {e}')
				return _wrap_result(action, {'index': index, 'value': ''})

		elif get_command == 'attributes':
			index = params['index']
			node = await bs.get_element_by_index(index)
			if node is None:
				return _wrap_result(action, {'error': f'Element index {index} not found - page may have changed'})
			attrs = node.attributes or {}
			return _wrap_result(action, {'index': index, 'attributes': dict(attrs)})

		elif get_command == 'bbox':
			index = params['index']
			node = await bs.get_element_by_index(index)
			if node is None:
				return _wrap_result(action, {'error': f'Element index {index} not found - page may have changed'})

			try:
				cdp_session = await bs.cdp_client_for_node(node)
				box_result = await cdp_session.cdp_client.send.DOM.getBoxModel(
					params={'backendNodeId': node.backend_node_id},
					session_id=cdp_session.session_id,
				)

				model = box_result['model']  # type: ignore[index]
				content = model.get('content', [])  # type: ignore[union-attr]

				if len(content) >= 8:
					x = min(content[0], content[2], content[4], content[6])
					y = min(content[1], content[3], content[5], content[7])
					width = max(content[0], content[2], content[4], content[6]) - x
					height = max(content[1], content[3], content[5], content[7]) - y
					return _wrap_result(action, {'index': index, 'bbox': {'x': x, 'y': y, 'width': width, 'height': height}})
				else:
					return _wrap_result(action, {'index': index, 'bbox': {}})
			except Exception as e:
				logger.error(f'Failed to get element bbox: {e}')
				return _wrap_result(action, {'index': index, 'bbox': {}})

		return _wrap_result(action, {'error': 'Invalid get command. Use: title, html, text, value, attributes, bbox'})

	elif action == 'record':
		# CLIBrowserSession skips watchdogs by default — attach RecordingWatchdog lazily on first use.
		watchdog = getattr(bs, '_recording_watchdog', None)
		if watchdog is None:
			from browser_use.browser.watchdogs.recording_watchdog import RecordingWatchdog

			RecordingWatchdog.model_rebuild()
			watchdog = RecordingWatchdog(event_bus=bs.event_bus, browser_session=bs)
			watchdog.attach_to_session()
			bs._recording_watchdog = watchdog

		record_command = params.get('record_command')

		if record_command == 'start':
			path = params.get('path')
			if not path:
				return _wrap_result(action, {'error': 'Usage: record start <output-path>'})
			if watchdog.is_recording:
				return _wrap_result(action, {'error': 'Recording already in progress. Call `record stop` first.'})

			output_path = Path(path).expanduser()
			try:
				output_path.parent.mkdir(parents=True, exist_ok=True)
			except OSError as e:
				return _wrap_result(action, {'error': f'Cannot create output directory {output_path.parent}: {e}'})

			framerate = params.get('framerate')
			try:
				saved = await watchdog.start_recording(output_path, framerate=framerate)
			except RuntimeError as e:
				return _wrap_result(action, {'error': str(e)})
			return _wrap_result(action, {'recording': True, 'path': str(saved)})

		elif record_command == 'stop':
			if not watchdog.is_recording:
				return _wrap_result(action, {'error': 'No recording in progress'})
			saved = await watchdog.stop_recording()
			if saved is None:
				return _wrap_result(action, {'error': 'No recording in progress'})
			return _wrap_result(action, {'_raw_text': str(saved)})

		elif record_command == 'status':
			recorder = watchdog._recorder
			if recorder is None:
				return _wrap_result(action, {'recording': False})
			return _wrap_result(
				action,
				{
					'recording': True,
					'path': str(recorder.output_path),
					'framerate': recorder.framerate,
					'size': {'width': recorder.size['width'], 'height': recorder.size['height']},
				},
			)

		return _wrap_result(action, {'error': 'Invalid record command. Use: start <path>, stop, status'})

	raise ValueError(f'Unknown browser action: {action}')
