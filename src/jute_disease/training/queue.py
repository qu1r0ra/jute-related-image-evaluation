"""Deadline checks for the training queue."""

import math
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def check_deadline(config: dict[str, Any], now: str) -> dict[str, Any]:
    """Check a forecast using a default and minimum two-hour reserve.

    The cutoff, its timezone and verification evidence must be explicit.
    A passing forecast does not start or admit a run.
    """
    if (
        not config.get("cutoff")
        or config.get("verified") is not True
        or not isinstance(config.get("verification"), str)
        or not config["verification"].strip()
        or not config.get("timezone")
    ):
        raise ValueError(
            "an explicit verified cutoff, timezone and evidence are required"
        )
    try:
        zone = ZoneInfo(config["timezone"])
        cutoff = datetime.fromisoformat(config["cutoff"])
        current = datetime.fromisoformat(now)
    except (TypeError, ValueError, ZoneInfoNotFoundError) as error:
        raise ValueError("invalid cutoff, timezone or clock") from error
    if cutoff.utcoffset() is None or current.utcoffset() is None:
        raise ValueError("cutoff and clock must include a UTC offset")
    if cutoff.utcoffset() != cutoff.astimezone(zone).utcoffset():
        raise ValueError("cutoff offset does not match timezone")
    seconds = []
    for key in ("mandatory_seconds", "optional_seconds", "finalization_seconds"):
        value = config.get(key, 2 * 3600 if key == "finalization_seconds" else None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a nonnegative finite number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a nonnegative finite number")
        seconds.append(value)
    mandatory, optional, reserve = seconds
    if reserve < 2 * 3600:
        raise ValueError("finalization reserve must be at least two hours")
    projected = current.astimezone(UTC) + timedelta(
        seconds=mandatory + optional + reserve
    )
    if projected > cutoff.astimezone(UTC):
        raise ValueError("forecast would consume the finalization reserve")
    return {"forecast_fits": True, "projected_end": projected.isoformat()}
