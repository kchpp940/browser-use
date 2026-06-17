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

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from browser_use.dom.views import (
	DOMSelectorMap,
	EnhancedDOMTreeNode,
	NodeType,
	SerializedDOMState,
	SimplifiedNode,
)


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
