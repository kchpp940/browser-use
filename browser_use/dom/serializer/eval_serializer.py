# @file purpose: Concise evaluation serializer for DOM trees - optimized for LLM query writing

from browser_use.dom.utils import cap_text_length
from browser_use.dom.views import (
	EnhancedDOMTreeNode,
	NodeType,
	SimplifiedNode,
)

# Critical attributes for query writing and form interaction
# NOTE: Removed 'id' and 'class' to force more robust structural selectors
EVAL_KEY_ATTRIBUTES = [
	'id',  # Removed - can have special chars, forces structural selectors
	'class',  # Removed - can have special chars like +, forces structural selectors
	'name',
	'type',
	'placeholder',
	'aria-label',
	'role',
	'value',
	# 'href',
	'data-testid',
	'alt',  # for images
	'title',  # useful for tooltips/link context
	# State attributes (critical for form interaction)
	'checked',
	'selected',
	'disabled',
	'required',
	'readonly',
	# ARIA states
	'aria-expanded',
	'aria-pressed',
	'aria-checked',
	'aria-selected',
	'aria-invalid',
	# Validation attributes (help agents avoid brute force)
	'pattern',
	'min',
	'max',
	'minlength',
	'maxlength',
	'step',
	'aria-valuemin',
	'aria-valuemax',
	'aria-valuenow',
]

# Semantic elements that should always be shown
SEMANTIC_ELEMENTS = {
	'html',  # Always show document root
	'body',  # Always show body
	'h1',
	'h2',
	'h3',
	'h4',
	'h5',
	'h6',
	'a',
	'button',
	'input',
	'textarea',
	'select',
	'form',
	'label',
	'nav',
	'header',
	'footer',
	'main',
	'article',
	'section',
	'table',
	'thead',
	'tbody',
	'tr',
	'th',
	'td',
	'ul',
	'ol',
	'li',
	'img',
	'iframe',
	'video',
	'audio',
}

# Container elements that can be collapsed if they only wrap one child
COLLAPSIBLE_CONTAINERS = {'div', 'span', 'section', 'article'}

# SVG child elements to skip (decorative only, no interaction value)
SVG_ELEMENTS = {
	'path',
	'rect',
	'g',
	'circle',
	'ellipse',
	'line',
	'polyline',
	'polygon',
	'use',
	'defs',
	'clipPath',
	'mask',
	'pattern',
	'image',
	'text',
	'tspan',
}


class DOMEvalSerializer:
	"""Ultra-concise DOM serializer for quick LLM query writing.

	NOTE: This serializer operates on the SAME SimplifiedNode tree as the main
	DOMTreeSerializer. This ensures interactive element indices (backend_node_id)
	are perfectly consistent with the selector_map used for action execution.

	The eval serializer shows MORE context (all semantic elements, not just
	interactive ones) but uses the same element set and the same interactivity
	source of truth (node.is_interactive).
	"""

	@staticmethod
	def serialize_tree(node: SimplifiedNode | None, include_attributes: list[str], depth: int = 0) -> str:
		"""
		Serialize complete DOM tree structure for LLM understanding.

		Strategy:
		- Uses SimplifiedNode tree as the single source of truth (consistent with selector_map)
		- Shows semantic elements for context, not just interactive ones
		- Interactive elements show full attributes + [i_backend_node_id]
		- Self-closing tags only (no closing tags)
		"""
		if not node:
			return ''

		# Skip excluded nodes but process children (consistent with DOMTreeSerializer)
		if hasattr(node, 'excluded_by_parent') and node.excluded_by_parent:
			return DOMEvalSerializer._serialize_children(node, include_attributes, depth)

		# Skip nodes marked as should_display=False (consistent with DOMTreeSerializer)
		if not node.should_display:
			return DOMEvalSerializer._serialize_children(node, include_attributes, depth)

		formatted_text = []
		depth_str = depth * '\t'

		if node.original_node.node_type == NodeType.ELEMENT_NODE:
			tag = node.original_node.tag_name.lower()

			# Special handling for iframes - show them with their content
			# NOTE: Content comes from node.children (SimplifiedNode), NOT raw content_document
			if tag in ['iframe', 'frame']:
				return DOMEvalSerializer._serialize_iframe(node, include_attributes, depth)

			# Special handling for SVG elements - show the tag but collapse children
			# SVG child elements are already filtered out of SimplifiedNode tree by _create_simplified_tree
			if tag == 'svg':
				line = f'{depth_str}'
				# Add [i_X] for interactive SVG elements only
				if node.is_interactive:
					line += f'[i_{node.original_node.backend_node_id}] '
				line += '<svg'
				attributes_str = DOMEvalSerializer._build_compact_attributes(node.original_node)
				if attributes_str:
					line += f' {attributes_str}'
				line += ' /> <!-- SVG content collapsed -->'
				return line

			# Build compact attributes string
			attributes_str = DOMEvalSerializer._build_compact_attributes(node.original_node)

			# Decide if this element should be shown
			is_semantic = tag in SEMANTIC_ELEMENTS
			has_useful_attrs = bool(attributes_str)
			has_text_content = DOMEvalSerializer._has_direct_text(node)
			has_children = len(node.children) > 0

			# Build compact element representation
			line = f'{depth_str}'
			# Add backend node ID notation - [i_X] for interactive elements only
			# Uses SAME backend_node_id as DOMTreeSerializer and selector_map
			if node.is_interactive:
				line += f'[i_{node.original_node.backend_node_id}] '
			line += f'<{tag}'

			if attributes_str:
				line += f' {attributes_str}'

			# Add scroll info if element is scrollable
			if node.original_node.should_show_scroll_info:
				scroll_text = node.original_node.get_scroll_info_text()
				if scroll_text:
					line += f' scroll="{scroll_text}"'

			# Add inline text if present (keep it on same line for compactness)
			inline_text = DOMEvalSerializer._get_inline_text(node)

			# Container elements that always show children
			container_tags = {'html', 'body', 'div', 'main', 'section', 'article', 'aside', 'header', 'footer', 'nav'}
			is_container = tag in container_tags

			if inline_text and not is_container:
				line += f'>{inline_text}'
			else:
				line += ' />'

			formatted_text.append(line)

			# Process children (always for containers, only if no inline_text for others)
			if has_children and (is_container or not inline_text):
				children_text = DOMEvalSerializer._serialize_children(node, include_attributes, depth + 1)
				if children_text:
					formatted_text.append(children_text)

		elif node.original_node.node_type == NodeType.TEXT_NODE:
			# Text nodes are handled inline with their parent
			pass

		elif node.original_node.node_type == NodeType.DOCUMENT_FRAGMENT_NODE:
			# Shadow DOM - show children directly with minimal marker
			if node.children:
				formatted_text.append(f'{depth_str}#shadow')
				children_text = DOMEvalSerializer._serialize_children(node, include_attributes, depth + 1)
				if children_text:
					formatted_text.append(children_text)

		return '\n'.join(formatted_text)

	@staticmethod
	def _serialize_children(node: SimplifiedNode, include_attributes: list[str], depth: int) -> str:
		"""Helper to serialize all children of a node."""
		children_output = []

		# Check if parent is a list container (ul, ol)
		is_list_container = node.original_node.node_type == NodeType.ELEMENT_NODE and node.original_node.tag_name.lower() in [
			'ul',
			'ol',
		]

		# Track list items and consecutive links
		li_count = 0
		max_list_items = 50
		consecutive_link_count = 0
		max_consecutive_links = 50
		total_links_skipped = 0

		for child in node.children:
			# Get tag name for this child
			current_tag = None
			if child.original_node.node_type == NodeType.ELEMENT_NODE:
				current_tag = child.original_node.tag_name.lower()

			# If we're in a list container and this child is an li element
			if is_list_container and current_tag == 'li':
				li_count += 1
				# Skip li elements after the 50th one
				if li_count > max_list_items:
					continue

			# Track consecutive anchor tags (links)
			if current_tag == 'a':
				consecutive_link_count += 1
				# Skip links after the 50th consecutive one
				if consecutive_link_count > max_consecutive_links:
					total_links_skipped += 1
					continue
			else:
				# Reset counter when we hit a non-link element
				# But first add truncation message if we skipped links
				if total_links_skipped > 0:
					depth_str = depth * '\t'
					children_output.append(f'{depth_str}... ({total_links_skipped} more links in this list)')
					total_links_skipped = 0
				consecutive_link_count = 0

			child_text = DOMEvalSerializer.serialize_tree(child, include_attributes, depth)
			if child_text:
				children_output.append(child_text)

		# Add truncation message if we skipped items at the end
		if is_list_container and li_count > max_list_items:
			depth_str = depth * '\t'
			children_output.append(
				f'{depth_str}... ({li_count - max_list_items} more items in this list (truncated) use evaluate to get more.'
			)

		# Add truncation message for links if we skipped any at the end
		if total_links_skipped > 0:
			depth_str = depth * '\t'
			children_output.append(
				f'{depth_str}... ({total_links_skipped} more links in this list) (truncated) use evaluate to get more.'
			)

		return '\n'.join(children_output)

	@staticmethod
	def _serialize_iframe(node: SimplifiedNode, include_attributes: list[str], depth: int) -> str:
		"""Handle iframe serialization with content from SimplifiedNode children.

		IMPORTANT: Uses node.children (SimplifiedNode tree) as the source of truth
		for iframe content, NOT node.original_node.content_document. This ensures:
		- Same element filtering as the main DOM tree
		- Same interactive element detection
		- Same indices as the selector_map
		"""
		formatted_text = []
		depth_str = depth * '\t'
		tag = node.original_node.tag_name.lower()

		# Build iframe element with key attributes
		attributes_str = DOMEvalSerializer._build_compact_attributes(node.original_node)
		line = f'{depth_str}<{tag}'
		if attributes_str:
			line += f' {attributes_str}'

		# Add scroll info for iframe content
		if node.original_node.should_show_scroll_info:
			scroll_text = node.original_node.get_scroll_info_text()
			if scroll_text:
				line += f' scroll="{scroll_text}"'

		line += ' />'
		formatted_text.append(line)

		# Serialize iframe content from SimplifiedNode children (single source of truth)
		if node.children:
			formatted_text.append(f'{depth_str}\t#iframe-content')
			children_text = DOMEvalSerializer._serialize_children(node, include_attributes, depth + 2)
			if children_text:
				formatted_text.append(children_text)

		return '\n'.join(formatted_text)

	@staticmethod
	def _build_compact_attributes(node: EnhancedDOMTreeNode) -> str:
		"""Build ultra-compact attributes string with only key attributes."""
		attrs = []

		# Prioritize attributes that help with query writing
		if node.attributes:
			for attr in EVAL_KEY_ATTRIBUTES:
				if attr in node.attributes:
					value = str(node.attributes[attr]).strip()
					if not value:
						continue

					# Special handling for different attributes
					if attr == 'class':
						# For class, limit to first 2 classes to save space
						classes = value.split()[:3]
						value = ' '.join(classes)
					elif attr == 'href':
						# For href, cap at 20 chars to save space
						value = cap_text_length(value, 80)
					else:
						# Cap at 25 chars for other attributes
						value = cap_text_length(value, 80)

					attrs.append(f'{attr}="{value}"')

		# Note: We intentionally don't add role from ax_node here because:
		# 1. If role is explicitly set in HTML, it's already captured above via EVAL_KEY_ATTRIBUTES
		# 2. Inferred roles from AX tree (like link, listitem, LineBreak) are redundant with the tag name
		# 3. This reduces noise - <a href="..." role="link"> is redundant, we already know <a> is a link

		return ' '.join(attrs)

	@staticmethod
	def _has_direct_text(node: SimplifiedNode) -> bool:
		"""Check if node has direct text children (not nested in other elements)."""
		for child in node.children:
			if child.original_node.node_type == NodeType.TEXT_NODE:
				text = child.original_node.node_value.strip() if child.original_node.node_value else ''
				if len(text) > 1:
					return True
		return False

	@staticmethod
	def _get_inline_text(node: SimplifiedNode) -> str:
		"""Get text content to display inline (max 40 chars)."""
		text_parts = []
		for child in node.children:
			if child.original_node.node_type == NodeType.TEXT_NODE:
				text = child.original_node.node_value.strip() if child.original_node.node_value else ''
				if text and len(text) > 1:
					text_parts.append(text)

		if not text_parts:
			return ''

		combined = ' '.join(text_parts)
		return cap_text_length(combined, 80)
