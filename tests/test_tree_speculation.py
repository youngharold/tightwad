"""Tests for tree-based speculative decoding."""

import pytest

from tightwad.speculation import DraftToken
from tightwad.tree_speculation import (
    PathVerificationResult,
    SpeculationTree,
    TreeNode,
    build_branching_tree,
    build_linear_tree,
    select_best_path,
)


def _d(tid: int, text: str = "") -> DraftToken:
    return DraftToken(token_id=tid, logprob=-0.1, text=text or f"t{tid}")


class TestTreeNode:
    def test_leaf(self):
        n = TreeNode(token=_d(1))
        assert n.is_leaf is True

    def test_not_leaf(self):
        child = TreeNode(token=_d(2))
        n = TreeNode(token=_d(1), children=[child])
        assert n.is_leaf is False

    def test_all_paths_leaf(self):
        n = TreeNode(token=_d(1))
        paths = n.all_paths()
        assert len(paths) == 1
        assert paths[0][0].token_id == 1

    def test_all_paths_chain(self):
        c = TreeNode(token=_d(3))
        b = TreeNode(token=_d(2), children=[c])
        a = TreeNode(token=_d(1), children=[b])
        paths = a.all_paths()
        assert len(paths) == 1
        assert [t.token_id for t in paths[0]] == [1, 2, 3]

    def test_all_paths_branching(self):
        left = TreeNode(token=_d(2))
        right = TreeNode(token=_d(3))
        root = TreeNode(token=_d(1), children=[left, right])
        paths = root.all_paths()
        assert len(paths) == 2
        assert [t.token_id for t in paths[0]] == [1, 2]
        assert [t.token_id for t in paths[1]] == [1, 3]


class TestBuildLinearTree:
    def test_empty(self):
        tree = build_linear_tree([])
        assert tree.n_paths == 0

    def test_single_token(self):
        tree = build_linear_tree([_d(1)])
        assert tree.n_paths == 1
        assert tree.longest_path_length == 1

    def test_multiple_tokens(self):
        tree = build_linear_tree([_d(1), _d(2), _d(3)])
        assert tree.n_paths == 1
        assert tree.longest_path_length == 3
        paths = tree.all_paths()
        assert [t.token_id for t in paths[0]] == [1, 2, 3]


class TestBuildBranchingTree:
    def test_single_draft(self):
        tree = build_branching_tree([[_d(1), _d(2), _d(3)]])
        assert tree.n_paths == 1

    def test_identical_drafts(self):
        drafts = [
            [_d(1), _d(2), _d(3)],
            [_d(1), _d(2), _d(3)],
        ]
        tree = build_branching_tree(drafts)
        assert tree.n_paths >= 1
        # All paths should contain [1, 2, 3]
        for path in tree.all_paths():
            assert path[0].token_id == 1

    def test_diverging_drafts(self):
        drafts = [
            [_d(1), _d(2), _d(10)],
            [_d(1), _d(2), _d(20)],
        ]
        tree = build_branching_tree(drafts)
        # Common prefix [1, 2], then branches [10] and [20]
        assert tree.n_paths == 2
        paths = tree.all_paths()
        # Both paths start with 1, 2
        for path in paths:
            assert path[0].token_id == 1
            assert path[1].token_id == 2

    def test_immediate_divergence(self):
        drafts = [
            [_d(10), _d(11)],
            [_d(20), _d(21)],
        ]
        tree = build_branching_tree(drafts)
        assert tree.n_paths == 2

    def test_max_branches_limits(self):
        drafts = [
            [_d(i), _d(i + 100)] for i in range(10)
        ]
        tree = build_branching_tree(drafts, max_branches=3)
        assert tree.n_paths <= 3

    def test_empty(self):
        tree = build_branching_tree([])
        assert tree.n_paths == 0


class TestSpeculationTree:
    def test_longest_path(self):
        tree = build_branching_tree([
            [_d(1), _d(2), _d(3), _d(4)],
            [_d(1), _d(2), _d(5)],
        ])
        assert tree.longest_path_length == 4


class TestSelectBestPath:
    def test_selects_longest(self):
        results = [
            PathVerificationResult(path=[_d(1)], accepted_count=1, full_match=False),
            PathVerificationResult(path=[_d(1), _d(2), _d(3)], accepted_count=3, full_match=True),
            PathVerificationResult(path=[_d(1), _d(2)], accepted_count=2, full_match=True),
        ]
        best = select_best_path(results)
        assert best.accepted_count == 3

    def test_prefers_full_match(self):
        results = [
            PathVerificationResult(path=[_d(1), _d(2)], accepted_count=2, full_match=True),
            PathVerificationResult(path=[_d(1), _d(2)], accepted_count=2, full_match=False),
        ]
        best = select_best_path(results)
        assert best.full_match is True

    def test_empty(self):
        best = select_best_path([])
        assert best.accepted_count == 0
