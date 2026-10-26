"""Dependency-free summaries for training and model inspection."""

from __future__ import annotations


def summarize_history(history) -> str:
    values = getattr(history, "values", history)
    lines = ["EasyMoDD training summary"]
    for name, points in values.items():
        if points:
            lines.append(f"{name}: start={points[0]:.6g}, final={points[-1]:.6g}, points={len(points)}")
    return "\n".join(lines)


def summarize_model(model) -> dict[str, int]:
    parameters = list(model.parameters()) if hasattr(model, "parameters") else []
    return {
        "parameter_groups": len(parameters),
        "parameters": sum(parameter.value.size for parameter in parameters),
    }
