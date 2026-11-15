"""Agents package for Project 7 — Decentralized Insurance Fraud Detection."""

from agents.base import (
    SEVERITY_CLASSES,
    SEVERITY_TO_INDEX,
    INDEX_TO_SEVERITY,
    TABULAR_SEVERITY_MAP,
    AgentMessage,
    BaseAgent,
    compute_severity_mismatch,
    normalize_severity_label,
)

__all__ = [
    "SEVERITY_CLASSES",
    "SEVERITY_TO_INDEX",
    "INDEX_TO_SEVERITY",
    "TABULAR_SEVERITY_MAP",
    "AgentMessage",
    "BaseAgent",
    "compute_severity_mismatch",
    "normalize_severity_label",
]
