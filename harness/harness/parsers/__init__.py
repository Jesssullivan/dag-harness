"""
Parsers for extracting metadata from Ansible roles and related files.
"""

from harness.parsers.ansible_parser import (
    AnsibleRoleParser,
    CredentialRef,
    RoleInfo,
)

__all__ = [
    "AnsibleRoleParser",
    "CredentialRef",
    "RoleInfo",
]
