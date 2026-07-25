"""Alert sinks for the drift gate.

`build_alerter` picks a Slack webhook when one is configured and falls back to
the console otherwise, so local runs and CI stay silent-but-observable without
needing a secret.
"""

import json
import urllib.error
import urllib.request
from typing import Protocol

from .drift import DriftReport

_SLACK_TIMEOUT_SECONDS = 5


class Alerter(Protocol):
    """Anything that can deliver a drift message."""

    def send(self, message: str) -> bool: ...


class ConsoleAlerter:
    """Prints alerts and records them — the default outside production."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, message: str) -> bool:
        self.sent.append(message)
        print(f"[drift-alert] {message}")
        return True


class SlackAlerter:
    """Posts alerts to a Slack incoming webhook."""

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url
        self.sent: list[str] = []

    def send(self, message: str) -> bool:
        payload = json.dumps({"text": message}).encode("utf-8")
        request = urllib.request.Request(
            self.webhook_url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=_SLACK_TIMEOUT_SECONDS) as response:
                delivered = 200 <= response.status < 300
        except (urllib.error.URLError, TimeoutError) as error:
            # Never let a webhook outage fail the training or drift job.
            print(f"[drift-alert] Slack delivery failed: {error}")
            return False
        if delivered:
            self.sent.append(message)
        return delivered


def build_alerter(webhook_url: str = "") -> Alerter:
    """Slack when a webhook is configured, console otherwise."""
    return SlackAlerter(webhook_url) if webhook_url else ConsoleAlerter()


def format_drift_message(report: DriftReport, model_name: str) -> str:
    details = ", ".join(
        f"{f.feature} (PSI {f.psi:.3f}, KS {f.ks:.3f})"
        for f in report.features
        if f.drifted
    )
    return (
        f":warning: Drift detected for *{model_name}* — "
        f"{len(report.drifted_features)} of {len(report.features)} features shifted. {details}"
    )


def alert_on_drift(report: DriftReport, alerter: Alerter, model_name: str) -> bool:
    """Send an alert only when drift is present. Returns whether one was sent."""
    if not report.has_drift:
        return False
    return alerter.send(format_drift_message(report, model_name))
