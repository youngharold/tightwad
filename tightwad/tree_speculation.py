"""Tree-based speculative decoding (OPT-Tree style).

Instead of generating a single linear sequence of draft tokens, the draft
model explores multiple continuation branches at high-uncertainty positions.
The proxy verifies each branch against the target, accepting the longest
valid path.

This module provides the tree construction and path selection logic.  The
actual verification uses standard completion requests since llama.cpp does
not yet support tree attention natively.  When tree attention becomes
available, this module can switch to single-pass tree verification.

Current approach: expand K branches at each uncertainty point, verify the
top candidates via parallel completion requests, accept the longest match.

Limitations without native tree attention:
- Each branch requires a separate verify call (not a single batch)
- Still faster than linear when branching saves re-drafting on rejection
- Best for scenarios where acceptance is moderate (40-70%) — high enough
  that some branches pass, low enough that linear wastes tokens
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .speculation import DraftToken

logger = logging.getLogger("tightwad.tree_speculation")


# ---------------------------------------------------------------------------
# Tree data structures
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A node in the speculation tree."""

    token: DraftToken
    children: list[TreeNode] = field(default_factory=list)
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def path_tokens(self) -> list[DraftToken]:
        """Return tokens from root to this node."""
        return [self.token]

    def all_paths(self) -> list[list[DraftToken]]:
        """Return all root-to-leaf paths in this subtree."""
        if self.is_leaf:
            return [[self.token]]
        paths = []
        for child in self.children:
            for child_path in child.all_paths():
                paths.append([self.token] + child_path)
        return paths


@dataclass
class SpeculationTree:
    """A tree of speculative draft token sequences."""

    roots: list[TreeNode] = field(default_factory=list)
    total_nodes: int = 0
    branch_points: int = 0

    def all_paths(self) -> list[list[DraftToken]]:
        """Return all root-to-leaf paths across all root nodes."""
        paths = []
        for root in self.roots:
            paths.extend(root.all_paths())
        return paths

    @property
    def longest_path_length(self) -> int:
        paths = self.all_paths()
        return max(len(p) for p in paths) if paths else 0

    @property
    def n_paths(self) -> int:
        return len(self.all_paths())


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------


def build_linear_tree(tokens: list[DraftToken]) -> SpeculationTree:
    """Build a degenerate tree (single path) from a linear draft sequence.

    This is equivalent to standard speculation — no branching.
    """
    if not tokens:
        return SpeculationTree()

    tree = SpeculationTree(total_nodes=len(tokens))

    # Build chain: token[0] → token[1] → ... → token[n-1]
    nodes = [TreeNode(token=t, depth=i) for i, t in enumerate(tokens)]
    for i in range(len(nodes) - 1):
        nodes[i].children = [nodes[i + 1]]

    tree.roots = [nodes[0]]
    return tree


def build_branching_tree(
    drafts: list[list[DraftToken]],
    max_branches: int = 3,
) -> SpeculationTree:
    """Build a tree from multiple draft sequences (e.g., from multiple drafters).

    Finds the longest common prefix, then branches at the first divergence
    point. Each unique continuation becomes a branch.

    Parameters
    ----------
    drafts:
        Multiple draft sequences (e.g., from consensus mode's all_drafts).
    max_branches:
        Maximum branches to keep at each divergence point.
    """
    if not drafts:
        return SpeculationTree()

    if len(drafts) == 1:
        return build_linear_tree(drafts[0])

    # Find common prefix length
    min_len = min(len(d) for d in drafts)
    prefix_len = 0
    for i in range(min_len):
        token_ids = {d[i].token_id for d in drafts}
        if len(token_ids) == 1:
            prefix_len = i + 1
        else:
            break

    # Build prefix as a linear chain
    if prefix_len == 0:
        # Immediate divergence — create branches from root
        branches: dict[int, list[list[DraftToken]]] = {}
        for draft in drafts:
            if draft:
                tid = draft[0].token_id
                if tid not in branches:
                    branches[tid] = []
                branches[tid].append(draft)

        tree = SpeculationTree(branch_points=1)
        for tid, branch_drafts in list(branches.items())[:max_branches]:
            # Use the first draft's tokens for this branch
            first = branch_drafts[0]
            subtree = build_linear_tree(first)
            tree.roots.extend(subtree.roots)
            tree.total_nodes += subtree.total_nodes
        return tree

    # Build common prefix
    prefix_tokens = drafts[0][:prefix_len]
    prefix_nodes = [TreeNode(token=t, depth=i) for i, t in enumerate(prefix_tokens)]
    for i in range(len(prefix_nodes) - 1):
        prefix_nodes[i].children = [prefix_nodes[i + 1]]

    # Branch at divergence point
    if prefix_len < min_len:
        branches = {}
        for draft in drafts:
            if prefix_len < len(draft):
                tid = draft[prefix_len].token_id
                if tid not in branches:
                    branches[tid] = draft[prefix_len:]

        branch_nodes = []
        for tid, remaining in list(branches.items())[:max_branches]:
            branch_chain = [
                TreeNode(token=t, depth=prefix_len + i)
                for i, t in enumerate(remaining)
            ]
            for i in range(len(branch_chain) - 1):
                branch_chain[i].children = [branch_chain[i + 1]]
            branch_nodes.append(branch_chain[0])

        prefix_nodes[-1].children = branch_nodes

    total_nodes = prefix_len + sum(
        len(d) - prefix_len for d in drafts[:max_branches]
        if len(d) > prefix_len
    )

    return SpeculationTree(
        roots=[prefix_nodes[0]],
        total_nodes=total_nodes,
        branch_points=1 if len(set(
            d[prefix_len].token_id for d in drafts
            if len(d) > prefix_len
        )) > 1 else 0,
    )


# ---------------------------------------------------------------------------
# Path selection
# ---------------------------------------------------------------------------


@dataclass
class PathVerificationResult:
    """Result of verifying a tree path against the target."""

    path: list[DraftToken]
    accepted_count: int
    full_match: bool


def select_best_path(results: list[PathVerificationResult]) -> PathVerificationResult:
    """Select the path with the most accepted tokens."""
    if not results:
        return PathVerificationResult(path=[], accepted_count=0, full_match=False)
    return max(results, key=lambda r: (r.accepted_count, r.full_match))
