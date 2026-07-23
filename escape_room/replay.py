from __future__ import annotations

from typing import Dict, List, Sequence


Attempt = Dict[str, object]


def filter_replay_attempts(
    attempts: Sequence[Attempt],
    outcome: str,
    ordering: str,
) -> List[Attempt]:
    """Return every attempt matching the selected library view."""
    filtered = [
        attempt
        for attempt in attempts
        if outcome == "All episodes"
        or (outcome == "Successful only" and bool(attempt["success"]))
        or (outcome == "Failed only" and not bool(attempt["success"]))
    ]

    if ordering == "Episode: newest first":
        filtered.sort(key=lambda attempt: int(attempt["episode"]), reverse=True)
    elif ordering == "Episode: oldest first":
        filtered.sort(key=lambda attempt: int(attempt["episode"]))
    else:
        filtered.sort(key=lambda attempt: float(attempt["reward"]), reverse=True)
    return filtered


def replay_library_rows(attempts: Sequence[Attempt], total_episodes: int) -> List[Dict[str, object]]:
    """Build lightweight rows for the complete episode table."""
    total_episodes = max(1, int(total_episodes))
    rows: List[Dict[str, object]] = []
    for attempt in attempts:
        episode = int(attempt["episode"])
        progress = min(1.0, max(0.0, episode / total_episodes))
        if progress <= 0.25:
            phase = "Early exploration"
        elif progress <= 0.75:
            phase = "Learning"
        else:
            phase = "Late policy"
        rows.append(
            {
                "Episode": episode,
                "Result": "SUCCESS" if bool(attempt["success"]) else "FAILED",
                "Reward": round(float(attempt["reward"]), 2),
                "Steps": int(attempt["steps"]),
                "Epsilon": round(float(attempt.get("epsilon", 0.0)), 4),
                "Training phase": phase,
            }
        )
    return rows
