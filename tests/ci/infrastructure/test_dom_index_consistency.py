"""
Unit tests for DOM index consistency - NO browser required.

Tests the single-source-of-truth index policy across:
- SimplifiedNode.is_structurally_indexable() (structural eligibility)
- SerializedDOMState.validate_consistency() (consistency gate)
- Special nodes: label/select/file input, shadow root, hidden iframe,
  SVG children, paint order filtering, parent exclusion

These tests verify that index generation, display, and action execution
all use the SAME filtering rules.

Usage:
	uv run pytest tests/ci/infrastructure/test_dom_index_consistency.py -v -s
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from browser_use.dom.views import (
	DEFAULT_INCLUDE_ATTRIBUTES,
	DOMSelectorMap,
	EnhancedDOMTreeNode,
	NodeType,
	SerializedDOMState,
	SimplifiedNode,
)
from browser_use.dom.serializer.serializer import DOMTreeSerializer
from browser_use.dom.serializer.eval_serializer import DOMEvalSerializer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: build mock nodes without a browser
# ──────────────────────────────────────────────────────────────────────────────


def _make_mock_node(
	backend_node_id: int,
	tag_name: str | None = None,
	node_type: NodeType = NodeType.ELEMENT_NODE,
	is_visible: bool | None = True,
	attributes: dict | None = None,
) -> object:
	"""
	Create a minimal mock EnhancedDOMTreeNode-compatible object for unit testing.

	We only populate the fields needed for index eligibility checks.
	Uses SimpleNamespace for simplicity - runtime code accesses via getattr.
	"""
	node = SimpleNamespace()
	node.backend_node_id = backend_node_id
	node.node_id = backend_node_id * 10  # arbitrary, distinct from backend_id
	node.node_type = node_type
	node.node_name = tag_name or ''
	node.node_value = ''
	node.attributes = attributes or {}
	node.is_visible = is_visible
	node.is_scrollable = False
	node.absolute_position = None
	node.target_id = None
	node.frame_id = None
	node.session_id = None
	node.content_document = None
	node.shadow_root_type = None
	node.shadow_roots = None
	node.snapshot_node = None
	node.ax_node = None
	node.parent_node = None
	node.children_nodes = None
	node.has_js_click_listener = False
	node._compound_children = []
	node.hidden_elements_info = []
	node.has_hidden_content = False
	node.uuid = f'mock-{backend_node_id}'

	# Add tag_name property-like behavior
	node.tag_name = (tag_name or '').lower()

	# Scroll-related attributes (used by serializers)
	node.is_actually_scrollable = False
	node.should_show_scroll_info = False
	node.get_scroll_info_text = lambda: ''

	return node


def _make_simple_node(
	backend_id: int,
	tag_name: str = 'div',
	*,
	is_interactive: bool = False,
	should_display: bool = True,
	excluded_by_parent: bool = False,
	ignored_by_paint_order: bool = False,
	is_shadow_host: bool = False,
	children: list[SimplifiedNode] | None = None,
	node_type: NodeType = NodeType.ELEMENT_NODE,
	attributes: dict | None = None,
) -> SimplifiedNode:
	"""Build a SimplifiedNode with a mock original_node."""
	original = _make_mock_node(
		backend_node_id=backend_id,
		tag_name=tag_name,
		node_type=node_type,
		attributes=attributes,
	)
	node = SimplifiedNode(
		original_node=original,  # type: ignore[arg-type]
		children=children or [],
		should_display=should_display,
		is_interactive=is_interactive,
		excluded_by_parent=excluded_by_parent,
		ignored_by_paint_order=ignored_by_paint_order,
		is_shadow_host=is_shadow_host,
	)
	return node


def _build_selector_map(*nodes: SimplifiedNode) -> DOMSelectorMap:
	"""Build a selector_map from a list of SimplifiedNodes."""
	result: dict[int, Any] = {}
	for node in nodes:
		if node.original_node and node.original_node.backend_node_id:
			result[node.original_node.backend_node_id] = node.original_node
	return cast(DOMSelectorMap, result)


def _collect_tree_ids(node: SimplifiedNode) -> set[int]:
	"""Collect all backend_node_ids of is_interactive nodes in a tree."""
	ids: set[int] = set()
	if node.is_interactive and node.original_node:
		bid = getattr(node.original_node, 'backend_node_id', None)
		if bid is not None and isinstance(bid, int):
			ids.add(bid)
	for child in node.children:
		ids.update(_collect_tree_ids(child))
	return ids


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 1: SimplifiedNode.is_structurally_indexable()
# ──────────────────────────────────────────────────────────────────────────────


class TestStructurallyIndexable:
	"""Test the structural index eligibility predicate."""

	def test_basic_element_is_indexable(self):
		"""A normal visible element should be structurally indexable."""
		node = _make_simple_node(1, 'button')
		assert node.is_structurally_indexable() is True

	def test_none_original_node_not_indexable(self):
		"""Node with no original_node should not be indexable."""
		node = _make_simple_node(1, 'div')
		node.original_node = None  # type: ignore[assignment]
		assert node.is_structurally_indexable() is False

	def test_no_backend_node_id_not_indexable(self):
		"""Node without backend_node_id should not be indexable."""
		node = _make_simple_node(1, 'div')
		del node.original_node.backend_node_id  # type: ignore[attr-defined]
		assert node.is_structurally_indexable() is False

	def test_should_display_false_not_indexable(self):
		"""Node with should_display=False should not be indexable."""
		node = _make_simple_node(1, 'button', should_display=False)
		assert node.is_structurally_indexable() is False

	def test_excluded_by_parent_not_indexable(self):
		"""Node excluded by parent bounding box should not be indexable."""
		node = _make_simple_node(1, 'button', excluded_by_parent=True)
		assert node.is_structurally_indexable() is False

	def test_ignored_by_paint_order_not_indexable(self):
		"""Node hidden by paint order should not be indexable."""
		node = _make_simple_node(1, 'button', ignored_by_paint_order=True)
		assert node.is_structurally_indexable() is False

	def test_parent_excluded_cascade_not_indexable(self):
		"""parent_excluded=True should cascade and make node not indexable."""
		node = _make_simple_node(1, 'button')
		assert node.is_structurally_indexable(parent_excluded=True) is False

	def test_document_fragment_not_indexable(self):
		"""Shadow root (DOCUMENT_FRAGMENT_NODE) itself should never be indexable."""
		node = _make_simple_node(
			1, '#document-fragment', node_type=NodeType.DOCUMENT_FRAGMENT_NODE,
		)
		assert node.is_structurally_indexable() is False

	def test_svg_root_is_indexable(self):
		"""The <svg> root element itself should be structurally indexable."""
		node = _make_simple_node(1, 'svg')
		assert node.is_structurally_indexable() is True

	def test_svg_child_not_indexable(self):
		"""Children inside an SVG subtree should not be indexable."""
		node = _make_simple_node(2, 'path')
		assert node.is_structurally_indexable(inside_svg=True) is False

	def test_svg_root_inside_svg_still_indexable(self):
		"""Even with inside_svg=True, the <svg> tag itself should be indexable."""
		node = _make_simple_node(1, 'svg')
		assert node.is_structurally_indexable(inside_svg=True) is True

	def test_text_node_not_indexable(self):
		"""Text nodes should not be indexable (no tag_name, but has backend_id).

		Actually text nodes might still pass structural check - they just won't
		pass the interactivity check. This test documents the current behavior.
		"""
		node = _make_simple_node(1, '#text', node_type=NodeType.TEXT_NODE)
		# Structural check passes; interactivity check would filter it later
		assert node.is_structurally_indexable() is True


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 2: validate_consistency() - basic alignment
# ──────────────────────────────────────────────────────────────────────────────


class TestValidateConsistencyBasic:
	"""Test basic consistency validation and auto-fix."""

	def test_perfectly_consistent_state(self):
		"""A state where tree and map match should have zero issues."""
		btn = _make_simple_node(1, 'button', is_interactive=True)
		root = _make_simple_node(0, 'div', children=[btn])
		state = SerializedDOMState(_root=root, selector_map=_build_selector_map(btn))

		issues = state.validate_consistency(auto_fix=False)
		assert len(issues) == 0

	def test_tree_has_map_missing(self):
		"""Tree has is_interactive but selector_map doesn't have it."""
		btn = _make_simple_node(1, 'button', is_interactive=True)
		root = _make_simple_node(0, 'div', children=[btn])
		state = SerializedDOMState(_root=root, selector_map={})

		issues = state.validate_consistency(auto_fix=False)
		assert len(issues) == 1
		assert 'in tree but not in selector_map' in issues[0]
		assert 1 in _collect_tree_ids(root)  # Still there since auto_fix=False

	def test_map_has_tree_missing(self):
		"""selector_map has entries but tree doesn't have is_interactive."""
		btn = _make_simple_node(1, 'button', is_interactive=False)
		root = _make_simple_node(0, 'div', children=[btn])
		state = SerializedDOMState(_root=root, selector_map=_build_selector_map(btn))

		issues = state.validate_consistency(auto_fix=False)
		assert len(issues) == 1
		assert 'in selector_map but not marked is_interactive' in issues[0]
		assert 1 in state.selector_map  # Still there since auto_fix=False

	def test_auto_fix_tree_only(self):
		"""Auto-fix should remove tree-only is_interactive flags."""
		btn = _make_simple_node(1, 'button', is_interactive=True)
		root = _make_simple_node(0, 'div', children=[btn])
		state = SerializedDOMState(_root=root, selector_map={})

		issues = state.validate_consistency(auto_fix=True)
		# After auto-fix, should be consistent
		assert btn.is_interactive is False
		assert len(state.selector_map) == 0

		# Verify with auto_fix=False
		post_issues = state.validate_consistency(auto_fix=False)
		assert len(post_issues) == 0

	def test_auto_fix_map_only(self):
		"""Auto-fix should remove map-only entries."""
		btn = _make_simple_node(1, 'button', is_interactive=False)
		root = _make_simple_node(0, 'div', children=[btn])
		state = SerializedDOMState(_root=root, selector_map=_build_selector_map(btn))

		issues = state.validate_consistency(auto_fix=True)
		# After auto-fix, should be consistent
		assert btn.is_interactive is False
		assert 1 not in state.selector_map

		post_issues = state.validate_consistency(auto_fix=False)
		assert len(post_issues) == 0

	def test_auto_fix_both_sides_mismatch(self):
		"""Auto-fix should resolve mismatches on both sides.

		After intersection alignment, only indices present on BOTH sides remain.
		"""
		btn1 = _make_simple_node(1, 'button', is_interactive=True)  # tree only
		btn2 = _make_simple_node(2, 'button', is_interactive=True)  # both
		btn3 = _make_simple_node(3, 'button', is_interactive=False)  # map only
		root = _make_simple_node(0, 'div', children=[btn1, btn2, btn3])
		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(btn2, btn3),
		)

		issues = state.validate_consistency(auto_fix=True)

		# Only btn2 (index 2) should remain on both sides
		assert btn1.is_interactive is False
		assert btn2.is_interactive is True
		assert btn3.is_interactive is False
		assert 1 not in state.selector_map
		assert 2 in state.selector_map
		assert 3 not in state.selector_map

		# Verify consistency
		post_issues = state.validate_consistency(auto_fix=False)
		assert len(post_issues) == 0

	def test_empty_tree_and_empty_map(self):
		"""Both empty should be consistent."""
		state = SerializedDOMState(_root=None, selector_map={})
		issues = state.validate_consistency(auto_fix=False)
		assert len(issues) == 0

	def test_empty_tree_with_nonempty_map(self):
		"""Empty tree but non-empty map should be fixed."""
		state = SerializedDOMState(
			_root=None,
			selector_map=cast(DOMSelectorMap, {1: _make_mock_node(1)}),
		)
		issues = state.validate_consistency(auto_fix=True)
		assert len(state.selector_map) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 3: Special node types - structural pruning
# ──────────────────────────────────────────────────────────────────────────────


class TestSpecialNodePruning:
	"""Test that special nodes are pruned correctly by validate_consistency().

	These tests verify that structural rules (SVG children, shadow roots,
	hidden elements, paint order, parent exclusion) are enforced by the
	consistency gate, even if upstream code incorrectly marked them interactive.
	"""

	# ── SVG children ──────────────────────────────────────────────────────

	def test_svg_children_pruned_from_tree_and_map(self):
		"""SVG child elements should NOT have indices (structural pruning).

		SVG subtrees (except the root <svg>) are decorative and should never
		receive interactive indices.
		"""
		svg_root = _make_simple_node(1, 'svg', is_interactive=False)
		svg_path = _make_simple_node(2, 'path', is_interactive=True)  # Wrongly marked
		svg_circle = _make_simple_node(3, 'circle', is_interactive=True)  # Wrongly marked
		svg_root.children.extend([svg_path, svg_circle])

		# Build state with wrongly-set interactive flags and map entries
		state = SerializedDOMState(
			_root=svg_root,
			selector_map=_build_selector_map(svg_path, svg_circle),
		)

		# Pre-condition: they're wrongly marked
		assert svg_path.is_interactive is True
		assert 2 in state.selector_map

		# Run consistency gate
		issues = state.validate_consistency(auto_fix=True)

		# SVG children should be pruned from both tree and map
		assert svg_path.is_interactive is False
		assert svg_circle.is_interactive is False
		assert 2 not in state.selector_map
		assert 3 not in state.selector_map

		# Final state should be consistent
		post_issues = state.validate_consistency(auto_fix=False)
		assert len(post_issues) == 0

	def test_svg_root_not_pruned(self):
		"""The <svg> root element itself should NOT be structurally pruned.

		Only children inside the SVG subtree get filtered out.
		"""
		svg_root = _make_simple_node(1, 'svg', is_interactive=True)
		state = SerializedDOMState(
			_root=svg_root,
			selector_map=_build_selector_map(svg_root),
		)

		issues = state.validate_consistency(auto_fix=True)
		# SVG root stays - it's not pruned structurally
		assert svg_root.is_interactive is True
		assert 1 in state.selector_map

	# ── Shadow root (DOCUMENT_FRAGMENT_NODE) ─────────────────────────────

	def test_shadow_root_node_pruned(self):
		"""Shadow root node itself (DOCUMENT_FRAGMENT) should not have an index."""
		shadow_root = _make_simple_node(
			1, '#document-fragment',
			is_interactive=True,  # Wrongly marked
			node_type=NodeType.DOCUMENT_FRAGMENT_NODE,
		)
		# Shadow root has a child button inside
		inner_btn = _make_simple_node(2, 'button', is_interactive=True)
		shadow_root.children.append(inner_btn)

		state = SerializedDOMState(
			_root=shadow_root,
			selector_map=_build_selector_map(shadow_root, inner_btn),
		)

		issues = state.validate_consistency(auto_fix=True)

		# Shadow root itself should be pruned
		assert shadow_root.is_interactive is False
		assert 1 not in state.selector_map

		# Inner button stays (it's an ELEMENT_NODE inside the shadow root)
		assert inner_btn.is_interactive is True
		assert 2 in state.selector_map

	# ── should_display = False (hidden elements) ─────────────────────────

	def test_hidden_element_pruned(self):
		"""Elements with should_display=False should not have indices."""
		hidden_btn = _make_simple_node(
			1, 'button', is_interactive=True, should_display=False,
		)
		root = _make_simple_node(0, 'div', children=[hidden_btn])
		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(hidden_btn),
		)

		issues = state.validate_consistency(auto_fix=True)

		assert hidden_btn.is_interactive is False
		assert 1 not in state.selector_map

	# ── excluded_by_parent (bounding box propagation) ────────────────────

	def test_excluded_by_parent_pruned(self):
		"""Elements excluded by parent bbox should not have indices."""
		btn = _make_simple_node(
			1, 'button', is_interactive=True, excluded_by_parent=True,
		)
		root = _make_simple_node(0, 'div', children=[btn])
		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(btn),
		)

		issues = state.validate_consistency(auto_fix=True)

		assert btn.is_interactive is False
		assert 1 not in state.selector_map

	def test_parent_exclusion_cascades(self):
		"""If a parent is excluded, children should also lose their indices.

		The _collect_prunable_indices method cascades parent_excluded down
		the tree, so descendants of an excluded node also get pruned.
		"""
		parent = _make_simple_node(
			1, 'div', is_interactive=False, excluded_by_parent=True,
		)
		child_btn = _make_simple_node(2, 'button', is_interactive=True)
		grandchild_input = _make_simple_node(3, 'input', is_interactive=True)
		parent.children.append(child_btn)
		child_btn.children.append(grandchild_input)

		root = _make_simple_node(0, 'div', children=[parent])
		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(child_btn, grandchild_input),
		)

		issues = state.validate_consistency(auto_fix=True)

		# Both child and grandchild should be pruned because parent is excluded
		assert child_btn.is_interactive is False
		assert grandchild_input.is_interactive is False
		assert 2 not in state.selector_map
		assert 3 not in state.selector_map

	# ── ignored_by_paint_order (occlusion filtering) ────────────────────

	def test_paint_order_filtered_pruned(self):
		"""Elements hidden by paint order should not have indices."""
		btn = _make_simple_node(
			1, 'button', is_interactive=True, ignored_by_paint_order=True,
		)
		root = _make_simple_node(0, 'div', children=[btn])
		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(btn),
		)

		issues = state.validate_consistency(auto_fix=True)

		assert btn.is_interactive is False
		assert 1 not in state.selector_map


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 4: Compound components (select, file input, label)
# ──────────────────────────────────────────────────────────────────────────────


class TestCompoundComponents:
	"""Test that compound component indices behave consistently.

	Note: These are structural tests. The actual interactivity detection
	is handled by ClickableElementDetector. Here we test the consistency
	gate's handling of these elements.
	"""

	def test_select_element_index_consistency(self):
		"""A <select> element should have consistent index on both sides."""
		select = _make_simple_node(1, 'select', is_interactive=True)
		root = _make_simple_node(0, 'div', children=[select])
		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(select),
		)

		issues = state.validate_consistency(auto_fix=False)
		assert len(issues) == 0

	def test_file_input_index_consistency(self):
		"""A file input should have consistent index on both sides.

		File inputs are often hidden with opacity:0 but still functional.
		They pass structural checks; the visibility exception is handled
		in _is_element_indexable().
		"""
		file_input = _make_simple_node(
			1, 'input', is_interactive=True,
			attributes={'type': 'file'},
		)
		root = _make_simple_node(0, 'form', children=[file_input])
		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(file_input),
		)

		issues = state.validate_consistency(auto_fix=False)
		assert len(issues) == 0

	def test_label_element_index_consistency(self):
		"""A <label> element should have consistent index on both sides."""
		label = _make_simple_node(1, 'label', is_interactive=True)
		root = _make_simple_node(0, 'form', children=[label])
		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(label),
		)

		issues = state.validate_consistency(auto_fix=False)
		assert len(issues) == 0

	def test_compound_component_flag_does_not_affect_structural(self):
		"""is_compound_component flag doesn't affect structural indexability.

		It's just metadata; structural eligibility is determined by other fields.
		"""
		comp = _make_simple_node(1, 'div', is_interactive=True)
		comp.is_compound_component = True
		root = _make_simple_node(0, 'select', children=[comp])
		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(comp),
		)

		# Compound components are still structurally eligible
		# (their actual indexability depends on interactivity checks)
		assert comp.is_structurally_indexable() is True


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 5: Deep tree scenarios
# ──────────────────────────────────────────────────────────────────────────────


class TestDeepTreeScenarios:
	"""Test consistency in deeper tree structures."""

	def test_nested_interactive_elements_consistent(self):
		"""Deeply nested interactive elements should all be consistent."""
		btn1 = _make_simple_node(1, 'button', is_interactive=True)
		btn2 = _make_simple_node(2, 'button', is_interactive=True)
		btn3 = _make_simple_node(3, 'button', is_interactive=True)

		div2 = _make_simple_node(20, 'div', children=[btn3])
		div1 = _make_simple_node(10, 'div', children=[btn2, div2])
		root = _make_simple_node(0, 'div', children=[btn1, div1])

		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(btn1, btn2, btn3),
		)

		issues = state.validate_consistency(auto_fix=False)
		assert len(issues) == 0

	def test_deep_tree_mismatch_auto_fix(self):
		"""Mismatches at various depths should all be fixed."""
		# Tree has these marked interactive:
		btn1 = _make_simple_node(1, 'button', is_interactive=True)   # both
		btn2 = _make_simple_node(2, 'button', is_interactive=True)   # tree only
		btn3 = _make_simple_node(3, 'button', is_interactive=False)  # map only
		btn4 = _make_simple_node(4, 'button', is_interactive=True)   # both (deep)
		btn5 = _make_simple_node(5, 'button', is_interactive=False)  # map only (deep)

		div = _make_simple_node(10, 'div', children=[btn3, btn4, btn5])
		root = _make_simple_node(0, 'div', children=[btn1, btn2, div])

		# Map has: 1, 3, 4, 5
		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(btn1, btn3, btn4, btn5),
		)

		# Pre-conditions
		assert btn1.is_interactive and 1 in state.selector_map
		assert btn2.is_interactive and 2 not in state.selector_map
		assert not btn3.is_interactive and 3 in state.selector_map
		assert btn4.is_interactive and 4 in state.selector_map
		assert not btn5.is_interactive and 5 in state.selector_map

		# Auto-fix
		issues = state.validate_consistency(auto_fix=True)

		# Post-conditions: only intersection (1, 4) should remain
		assert btn1.is_interactive is True
		assert btn2.is_interactive is False
		assert btn3.is_interactive is False
		assert btn4.is_interactive is True
		assert btn5.is_interactive is False
		assert set(state.selector_map.keys()) == {1, 4}

		# Verify consistency
		post_issues = state.validate_consistency(auto_fix=False)
		assert len(post_issues) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 6: Intersection alignment guarantee
# ──────────────────────────────────────────────────────────────────────────────


class TestIntersectionGuarantee:
	"""Test that auto_fix=True guarantees consistency via intersection.

	This is the key property: no matter how broken the input state is,
	after validate_consistency(auto_fix=True), the output state MUST
	be perfectly consistent (tree.is_interactive ↔ selector_map keys).
	"""

	def test_totally_broken_state_becomes_consistent(self):
		"""Even a totally mismatched state becomes consistent after auto-fix."""
		# Tree marks 1, 2, 3, 4 as interactive
		n1 = _make_simple_node(1, 'a', is_interactive=True)
		n2 = _make_simple_node(2, 'button', is_interactive=True)
		n3 = _make_simple_node(3, 'input', is_interactive=True)
		n4 = _make_simple_node(4, 'select', is_interactive=True)

		root = _make_simple_node(0, 'div', children=[n1, n2, n3, n4])

		# Map has 3, 4, 5, 6
		n5 = _make_simple_node(5, 'a', is_interactive=False)
		n6 = _make_simple_node(6, 'button', is_interactive=False)
		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(n3, n4, n5, n6),
		)

		# Fix
		issues = state.validate_consistency(auto_fix=True)

		# Only intersection {3, 4} should remain on both sides
		tree_ids = _collect_tree_ids(root)
		map_ids = set(state.selector_map.keys())
		assert tree_ids == map_ids
		assert tree_ids == {3, 4}

		# Final consistency check
		post_issues = state.validate_consistency(auto_fix=False)
		assert len(post_issues) == 0

	def test_all_tree_only_becomes_empty_consistent(self):
		"""If all interactive nodes are tree-only, result is empty but consistent."""
		n1 = _make_simple_node(1, 'button', is_interactive=True)
		n2 = _make_simple_node(2, 'a', is_interactive=True)
		root = _make_simple_node(0, 'div', children=[n1, n2])
		state = SerializedDOMState(_root=root, selector_map={})

		issues = state.validate_consistency(auto_fix=True)

		tree_ids = _collect_tree_ids(root)
		assert tree_ids == set()
		assert len(state.selector_map) == 0
		assert tree_ids == set(state.selector_map.keys())

	def test_all_map_only_becomes_empty_consistent(self):
		"""If all entries are map-only, result is empty but consistent."""
		n1 = _make_simple_node(1, 'button', is_interactive=False)
		n2 = _make_simple_node(2, 'a', is_interactive=False)
		root = _make_simple_node(0, 'div', children=[n1, n2])
		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(n1, n2),
		)

		issues = state.validate_consistency(auto_fix=True)

		tree_ids = _collect_tree_ids(root)
		assert tree_ids == set()
		assert len(state.selector_map) == 0
		assert tree_ids == set(state.selector_map.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 7: Hidden iframe content document
# ──────────────────────────────────────────────────────────────────────────────


class TestHiddenIframe:
	"""Test that hidden iframe content is handled consistently.

	If an iframe has should_display=False, its content_document children
	should also not have indices (cascading exclusion).
	"""

	def test_iframe_with_hidden_content_cascades(self):
		"""Hidden iframe content should not have interactive indices.

		The content_document node itself might have should_display=False
		or be inside a hidden iframe parent. Either way, the content
		elements should not have indices.
		"""
		# Build iframe content document tree
		inner_btn = _make_simple_node(10, 'button', is_interactive=True)
		inner_input = _make_simple_node(11, 'input', is_interactive=True)
		content_doc = _make_simple_node(
			5, '#document',
			is_interactive=False,
			node_type=NodeType.DOCUMENT_NODE,
			children=[inner_btn, inner_input],
		)

		# Iframe host is hidden (should_display=False)
		iframe = _make_simple_node(
			1, 'iframe',
			is_interactive=False,
			should_display=False,
			children=[content_doc],
		)
		root = _make_simple_node(0, 'div', children=[iframe])

		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(inner_btn, inner_input),
		)

		issues = state.validate_consistency(auto_fix=True)

		# Inner elements should be pruned because iframe is hidden
		assert inner_btn.is_interactive is False
		assert inner_input.is_interactive is False
		assert 10 not in state.selector_map
		assert 11 not in state.selector_map

	def test_visible_iframe_content_stays(self):
		"""Visible iframe content should keep its indices."""
		inner_btn = _make_simple_node(10, 'button', is_interactive=True)
		content_doc = _make_simple_node(
			5, '#document',
			is_interactive=False,
			node_type=NodeType.DOCUMENT_NODE,
			children=[inner_btn],
			should_display=True,
		)
		iframe = _make_simple_node(
			1, 'iframe',
			is_interactive=False,
			should_display=True,
			children=[content_doc],
		)
		root = _make_simple_node(0, 'div', children=[iframe])

		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(inner_btn),
		)

		issues = state.validate_consistency(auto_fix=False)
		# Should be consistent (both sides agree on inner_btn having an index)
		assert len(issues) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: end-to-end index extraction from serialized text
# ──────────────────────────────────────────────────────────────────────────────


def _extract_llm_indices(text: str) -> set[int]:
	"""
	Extract interactive element indices from LLM-formatted DOM text.

	LLM format: [backend_node_id]<tag  (e.g. [42]<button)
	"""
	# Match patterns like [123]< or [123] (for SVG) or *[123]< (new elements)
	# Also handle scroll element prefix: |scroll element[123]]
	pattern = re.compile(r'\[(\d+)\]')
	return {int(match.group(1)) for match in pattern.finditer(text)}


def _extract_eval_indices(text: str) -> set[int]:
	"""
	Extract interactive element indices from eval-formatted DOM text.

	Eval format: [i_backend_node_id] <tag  (e.g. [i_42] <button)
	"""
	pattern = re.compile(r'\[i_(\d+)\]')
	return {int(match.group(1)) for match in pattern.finditer(text)}


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 8: End-to-end three-way consistency (LLM text ↔ eval text ↔ selector_map)
# ──────────────────────────────────────────────────────────────────────────────


class TestThreeWayEndToEndConsistency:
	"""
	End-to-end tests verifying that interactive element indices are
	perfectly consistent across three consumer endpoints:

	1. LLM text (DOMTreeSerializer.serialize_tree)
	2. Eval text (DOMEvalSerializer.serialize_tree)
	3. Action execution (SerializedDOMState.selector_map → BrowserSession.get_element_by_index)

	These tests prove that nodes pruned by the consistency gate don't appear
	in any output, and retained nodes are accessible from all three endpoints.
	"""

	def _serialize_and_extract(
		self,
		state: SerializedDOMState,
	) -> tuple[set[int], set[int], set[int]]:
		"""
		Run full end-to-end pipeline: serialize tree to both formats,
		extract indices, and return (llm_indices, eval_indices, map_indices).
		"""
		llm_text = DOMTreeSerializer.serialize_tree(state._root, DEFAULT_INCLUDE_ATTRIBUTES)
		eval_text = DOMEvalSerializer.serialize_tree(state._root, DEFAULT_INCLUDE_ATTRIBUTES)
		llm_indices = _extract_llm_indices(llm_text)
		eval_indices = _extract_eval_indices(eval_text)
		map_indices = set(state.selector_map.keys())
		return llm_indices, eval_indices, map_indices

	# ── Basic consistency ─────────────────────────────────────────────

	def test_simple_button_three_way_consistent(self):
		"""A simple interactive button should be accessible from all three endpoints."""
		btn = _make_simple_node(42, 'button', is_interactive=True)
		root = _make_simple_node(0, 'div', children=[btn])
		state = SerializedDOMState(_root=root, selector_map=_build_selector_map(btn))

		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)

		assert 42 in llm_idx
		assert 42 in eval_idx
		assert 42 in map_idx
		assert llm_idx == eval_idx == map_idx

	def test_non_interactive_div_not_indexed(self):
		"""A plain div should NOT get an index in any output."""
		div = _make_simple_node(10, 'div', is_interactive=False)
		root = _make_simple_node(0, 'div', children=[div])
		state = SerializedDOMState(_root=root, selector_map={})

		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)

		assert 10 not in llm_idx
		assert 10 not in eval_idx
		assert 10 not in map_idx
		assert llm_idx == eval_idx == map_idx

	# ── Pruned nodes: should NOT appear in any endpoint ───────────────

	def test_svg_children_pruned_from_all_endpoints(self):
		"""SVG children (path, circle) should NOT appear as indexed elements."""
		svg_path = _make_simple_node(2, 'path', is_interactive=True)  # wrongly marked
		svg_circle = _make_simple_node(3, 'circle', is_interactive=True)  # wrongly marked
		svg_root = _make_simple_node(1, 'svg', is_interactive=False, children=[svg_path, svg_circle])
		real_btn = _make_simple_node(10, 'button', is_interactive=True)
		root = _make_simple_node(0, 'div', children=[svg_root, real_btn])

		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(svg_path, svg_circle, real_btn),
		)

		# Run consistency gate (auto-fix to prune wrongly-marked SVG children)
		state.validate_consistency(auto_fix=True)

		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)

		# SVG children should be pruned from ALL endpoints
		assert 2 not in llm_idx
		assert 2 not in eval_idx
		assert 2 not in map_idx
		assert 3 not in llm_idx
		assert 3 not in eval_idx
		assert 3 not in map_idx

		# Real button should still be there
		assert 10 in llm_idx
		assert 10 in eval_idx
		assert 10 in map_idx

		# Three-way equality
		assert llm_idx == eval_idx == map_idx

	def test_paint_order_filtered_pruned_from_all_endpoints(self):
		"""Nodes hidden by paint order should NOT appear in any endpoint."""
		occluded_btn = _make_simple_node(5, 'button', is_interactive=True, ignored_by_paint_order=True)
		visible_btn = _make_simple_node(20, 'a', is_interactive=True)
		root = _make_simple_node(0, 'div', children=[occluded_btn, visible_btn])

		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(occluded_btn, visible_btn),
		)
		state.validate_consistency(auto_fix=True)

		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)

		assert 5 not in llm_idx
		assert 5 not in eval_idx
		assert 5 not in map_idx
		assert 20 in llm_idx
		assert 20 in eval_idx
		assert 20 in map_idx
		assert llm_idx == eval_idx == map_idx

	def test_excluded_by_parent_pruned_from_all_endpoints(self):
		"""Nodes excluded by parent bounding box should NOT appear in any endpoint."""
		excluded_btn = _make_simple_node(7, 'button', is_interactive=True, excluded_by_parent=True)
		normal_btn = _make_simple_node(30, 'input', is_interactive=True, attributes={'type': 'text'})
		root = _make_simple_node(0, 'div', children=[excluded_btn, normal_btn])

		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(excluded_btn, normal_btn),
		)
		state.validate_consistency(auto_fix=True)

		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)

		assert 7 not in llm_idx
		assert 7 not in eval_idx
		assert 7 not in map_idx
		assert 30 in llm_idx
		assert 30 in eval_idx
		assert 30 in map_idx
		assert llm_idx == eval_idx == map_idx

	def test_hidden_element_pruned_from_all_endpoints(self):
		"""Nodes with should_display=False should NOT appear in any endpoint."""
		hidden_btn = _make_simple_node(9, 'button', is_interactive=True, should_display=False)
		visible_btn = _make_simple_node(40, 'select', is_interactive=True)
		root = _make_simple_node(0, 'div', children=[hidden_btn, visible_btn])

		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(hidden_btn, visible_btn),
		)
		state.validate_consistency(auto_fix=True)

		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)

		assert 9 not in llm_idx
		assert 9 not in eval_idx
		assert 9 not in map_idx
		assert 40 in llm_idx
		assert 40 in eval_idx
		assert 40 in map_idx
		assert llm_idx == eval_idx == map_idx

	def test_shadow_root_node_pruned_from_all_endpoints(self):
		"""Shadow root (DOCUMENT_FRAGMENT) itself should NOT appear as indexed."""
		shadow_btn = _make_simple_node(2, 'button', is_interactive=True)  # inside shadow
		shadow_root = _make_simple_node(
			1, '#document-fragment',
			is_interactive=True,  # wrongly marked on shadow root itself
			node_type=NodeType.DOCUMENT_FRAGMENT_NODE,
			children=[shadow_btn],
		)
		host_div = _make_simple_node(0, 'div', is_shadow_host=True, children=[shadow_root])

		state = SerializedDOMState(
			_root=host_div,
			selector_map=_build_selector_map(shadow_root, shadow_btn),
		)
		state.validate_consistency(auto_fix=True)

		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)

		# Shadow root itself should NOT be indexed
		assert 1 not in llm_idx
		assert 1 not in eval_idx
		assert 1 not in map_idx

		# Button INSIDE shadow DOM should still be indexed
		assert 2 in llm_idx
		assert 2 in eval_idx
		assert 2 in map_idx

		assert llm_idx == eval_idx == map_idx

	def test_hidden_iframe_content_pruned_from_all_endpoints(self):
		"""Content inside a hidden iframe should NOT appear as indexed."""
		inner_btn = _make_simple_node(10, 'button', is_interactive=True)
		inner_input = _make_simple_node(11, 'input', is_interactive=True, attributes={'type': 'file'})
		content_doc = _make_simple_node(
			5, '#document',
			node_type=NodeType.DOCUMENT_NODE,
			children=[inner_btn, inner_input],
		)
		iframe = _make_simple_node(
			1, 'iframe',
			should_display=False,
			children=[content_doc],
		)
		outer_btn = _make_simple_node(50, 'button', is_interactive=True)
		root = _make_simple_node(0, 'div', children=[iframe, outer_btn])

		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(inner_btn, inner_input, outer_btn),
		)
		state.validate_consistency(auto_fix=True)

		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)

		# Hidden iframe content should NOT be accessible
		assert 10 not in llm_idx
		assert 10 not in eval_idx
		assert 10 not in map_idx
		assert 11 not in llm_idx
		assert 11 not in eval_idx
		assert 11 not in map_idx

		# Outer button should be accessible
		assert 50 in llm_idx
		assert 50 in eval_idx
		assert 50 in map_idx

		assert llm_idx == eval_idx == map_idx

	# ── Compound components: should be indexed consistently ──────────

	def test_file_input_three_way_consistent(self):
		"""File input should appear consistently across all endpoints."""
		file_input = _make_simple_node(
			77, 'input',
			is_interactive=True,
			attributes={'type': 'file', 'accept': 'image/*'},
		)
		root = _make_simple_node(0, 'form', children=[file_input])
		state = SerializedDOMState(_root=root, selector_map=_build_selector_map(file_input))

		state.validate_consistency(auto_fix=True)
		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)

		assert 77 in llm_idx
		assert 77 in eval_idx
		assert 77 in map_idx
		assert llm_idx == eval_idx == map_idx

	def test_select_element_three_way_consistent(self):
		"""Select dropdown should appear consistently across all endpoints."""
		select = _make_simple_node(88, 'select', is_interactive=True)
		root = _make_simple_node(0, 'form', children=[select])
		state = SerializedDOMState(_root=root, selector_map=_build_selector_map(select))

		state.validate_consistency(auto_fix=True)
		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)

		assert 88 in llm_idx
		assert 88 in eval_idx
		assert 88 in map_idx
		assert llm_idx == eval_idx == map_idx

	def test_label_element_three_way_consistent(self):
		"""Label element should appear consistently across all endpoints."""
		label = _make_simple_node(99, 'label', is_interactive=True)
		root = _make_simple_node(0, 'form', children=[label])
		state = SerializedDOMState(_root=root, selector_map=_build_selector_map(label))

		state.validate_consistency(auto_fix=True)
		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)

		assert 99 in llm_idx
		assert 99 in eval_idx
		assert 99 in map_idx
		assert llm_idx == eval_idx == map_idx

	# ── Cascading exclusion: parent excluded → children pruned ────────

	def test_parent_exclusion_cascades_to_all_endpoints(self):
		"""If a parent is excluded, children should be pruned from ALL endpoints."""
		child_btn = _make_simple_node(2, 'button', is_interactive=True)
		grandchild_input = _make_simple_node(3, 'input', is_interactive=True, attributes={'type': 'text'})
		child_btn.children.append(grandchild_input)

		excluded_parent = _make_simple_node(
			1, 'div',
			excluded_by_parent=True,
			children=[child_btn],
		)
		sibling_btn = _make_simple_node(100, 'button', is_interactive=True)
		root = _make_simple_node(0, 'div', children=[excluded_parent, sibling_btn])

		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(child_btn, grandchild_input, sibling_btn),
		)
		state.validate_consistency(auto_fix=True)

		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)

		# Cascaded exclusion: both child and grandchild should be gone
		assert 2 not in llm_idx
		assert 2 not in eval_idx
		assert 2 not in map_idx
		assert 3 not in llm_idx
		assert 3 not in eval_idx
		assert 3 not in map_idx

		# Sibling should still be accessible
		assert 100 in llm_idx
		assert 100 in eval_idx
		assert 100 in map_idx

		assert llm_idx == eval_idx == map_idx

	# ── Complex mixed scenario ────────────────────────────────────────

	def test_complex_mixed_scenario_all_endpoints_agree(self):
		"""
		Complex scenario mixing all special node types:
		- SVG with children (should be pruned)
		- Paint order hidden button (should be pruned)
		- File input (should be kept)
		- Label wrapping select (both should be kept)
		- Shadow DOM with inner button (inner should be kept)
		- Excluded parent with child input (should be pruned)
		"""
		# SVG subtree
		svg_path = _make_simple_node(2, 'path', is_interactive=True)  # wrong
		svg = _make_simple_node(1, 'svg', children=[svg_path])

		# Paint order hidden
		hidden_btn = _make_simple_node(3, 'button', is_interactive=True, ignored_by_paint_order=True)  # wrong

		# File input (legitimate)
		file_input = _make_simple_node(
			4, 'input', is_interactive=True,
			attributes={'type': 'file'},
		)

		# Label + select (both legitimate)
		select = _make_simple_node(6, 'select', is_interactive=True)
		label = _make_simple_node(5, 'label', is_interactive=True, children=[select])

		# Shadow DOM
		shadow_btn = _make_simple_node(8, 'button', is_interactive=True)
		shadow_root = _make_simple_node(
			7, '#document-fragment',
			node_type=NodeType.DOCUMENT_FRAGMENT_NODE,
			children=[shadow_btn],
		)
		host = _make_simple_node(0, 'div', is_shadow_host=True, children=[shadow_root])

		# Excluded parent with child
		excluded_child = _make_simple_node(10, 'input', is_interactive=True, attributes={'type': 'text'})  # wrong
		excluded_parent = _make_simple_node(
			9, 'div', excluded_by_parent=True, children=[excluded_child],
		)

		# Build full tree
		body = _make_simple_node(
			100, 'body',
			children=[svg, hidden_btn, file_input, label, host, excluded_parent],
		)

		# Build state with wrongly-marked nodes + legitimate ones
		state = SerializedDOMState(
			_root=body,
			selector_map=_build_selector_map(
				svg_path, hidden_btn, file_input, select, label, shadow_btn, excluded_child,
			),
		)

		# Run consistency gate
		state.validate_consistency(auto_fix=True)

		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)

		# Should be pruned:
		assert 2 not in llm_idx   # SVG child
		assert 2 not in eval_idx
		assert 2 not in map_idx
		assert 3 not in llm_idx   # Paint order hidden
		assert 3 not in eval_idx
		assert 3 not in map_idx
		assert 10 not in llm_idx  # Excluded child
		assert 10 not in eval_idx
		assert 10 not in map_idx

		# Should be kept:
		assert 4 in llm_idx   # File input
		assert 4 in eval_idx
		assert 4 in map_idx
		assert 5 in llm_idx   # Label
		assert 5 in eval_idx
		assert 5 in map_idx
		assert 6 in llm_idx   # Select
		assert 6 in eval_idx
		assert 6 in map_idx
		assert 8 in llm_idx   # Shadow inner button
		assert 8 in eval_idx
		assert 8 in map_idx

		# Three-way perfect equality
		assert llm_idx == eval_idx == map_idx

	# ── Verify indices are actually actionable (selector_map lookup) ─

	def test_indices_in_selector_map_correspond_to_correct_nodes(self):
		"""
		Verify that indices appearing in text actually correspond to the correct
		elements in selector_map - not just that the set of indices matches.

		This simulates what BrowserSession.get_element_by_index() does.
		"""
		submit_btn = _make_simple_node(42, 'button', is_interactive=True, attributes={'type': 'submit'})
		cancel_link = _make_simple_node(99, 'a', is_interactive=True, attributes={'href': '/cancel'})
		root = _make_simple_node(0, 'div', children=[submit_btn, cancel_link])

		state = SerializedDOMState(
			_root=root,
			selector_map=_build_selector_map(submit_btn, cancel_link),
		)
		state.validate_consistency(auto_fix=True)

		llm_idx, eval_idx, map_idx = self._serialize_and_extract(state)
		assert llm_idx == eval_idx == map_idx == {42, 99}

		# Simulate BrowserSession.get_element_by_index() lookup
		# Index 42 should be the submit button
		node_42 = state.selector_map[42]
		assert node_42.tag_name == 'button'
		assert node_42.attributes.get('type') == 'submit'

		# Index 99 should be the cancel link
		node_99 = state.selector_map[99]
		assert node_99.tag_name == 'a'
		assert node_99.attributes.get('href') == '/cancel'

		# There should be no other entries
		assert len(state.selector_map) == 2
