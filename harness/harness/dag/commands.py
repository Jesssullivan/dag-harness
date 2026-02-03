"""Typed payloads for HOTL Command resume pattern.

This module provides TypedDict payloads and factory functions for creating
LangGraph Command objects used in human-on-the-loop (HOTL) workflows.
"""

from datetime import datetime
from typing import TypedDict

from langgraph.types import Command


class ApprovalPayload(TypedDict):
    """Payload for approval/rejection commands in HOTL workflows.

    Attributes:
        approved: Whether the action was approved.
        reason: Human-readable explanation for the decision.
        reviewer: Identifier of the person who made the decision.
        reviewed_at: Timestamp when the review occurred.
    """

    approved: bool
    reason: str
    reviewer: str
    reviewed_at: datetime


class SkipPayload(TypedDict):
    """Payload for skip commands to jump to a specific node.

    Attributes:
        skip_to_node: The name of the node to skip to.
        reason: Human-readable explanation for why the skip is needed.
    """

    skip_to_node: str
    reason: str


def create_approval_command(
    approved: bool,
    reason: str,
    reviewer: str,
) -> Command:
    """Create a Command for approving or rejecting a workflow step.

    Args:
        approved: Whether to approve (True) or reject (False) the step.
        reason: Explanation for the approval/rejection decision.
        reviewer: Identifier of the reviewer making the decision.

    Returns:
        A Command object with the ApprovalPayload as the resume value.

    Example:
        >>> cmd = create_approval_command(True, "Looks good", "jsullivan2")
        >>> cmd.resume["approved"]
        True
    """
    payload: ApprovalPayload = {
        "approved": approved,
        "reason": reason,
        "reviewer": reviewer,
        "reviewed_at": datetime.now(),
    }
    return Command(resume=payload)


def create_skip_command(
    skip_to_node: str,
    reason: str,
) -> Command:
    """Create a Command to skip to a specific node in the workflow.

    Args:
        skip_to_node: The name of the target node to skip to.
        reason: Explanation for why the skip is being performed.

    Returns:
        A Command object with the SkipPayload as the resume value
        and goto set to the target node.

    Example:
        >>> cmd = create_skip_command("finalize", "Pre-approved, skipping review")
        >>> cmd.resume["skip_to_node"]
        'finalize'
    """
    payload: SkipPayload = {
        "skip_to_node": skip_to_node,
        "reason": reason,
    }
    return Command(resume=payload, goto=skip_to_node)
