"""Lightweight in-process alerting utilities."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence

from .config import AlertingConfig

LOGGER = logging.getLogger("neutryx.alerts")


@dataclass
class AlertMessage:
    """Structured alert message forwarded to notifiers."""

    name: str
    severity: str
    summary: str
    details: Mapping[str, object]
    happened_at: float = field(default_factory=lambda: time.time())


class BaseNotifier:
    """Notifier interface for alert delivery destinations."""

    def notify(self, message: AlertMessage) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class LoggingNotifier(BaseNotifier):
    """Default notifier that emits alerts via the standard logging system."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or LOGGER

    def notify(self, message: AlertMessage) -> None:
        log_method = (
            self._logger.error if message.severity.lower() == "critical" else self._logger.warning
        )
        log_method(
            "[%s] %s | details=%s",
            message.name,
            message.summary,
            dict(message.details),
        )


@dataclass
class AlertEvent:
    """Represents a recorded operational event used for alert evaluation."""

    timestamp: float
    kind: str
    success: bool
    duration: float
    attributes: Mapping[str, str]


class BaseAlertManager:
    """Interface for alert manager implementations."""

    def record_event(
        self,
        kind: str,
        success: bool,
        duration: float,
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        raise NotImplementedError  # pragma: no cover - interface


class NullAlertManager(BaseAlertManager):
    """Alert manager that ignores all events."""

    def record_event(
        self,
        kind: str,
        success: bool,
        duration: float,
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        return


class AlertManager(BaseAlertManager):
    """Evaluate alerting rules against operational events."""

    def __init__(
        self,
        config: AlertingConfig,
        notifiers: Optional[Sequence[BaseNotifier]] = None,
    ):
        self.config = config
        self._notifiers = list(notifiers or (LoggingNotifier(),))
        self._events: deque[AlertEvent] = deque()
        self._lock = threading.Lock()
        self._last_emitted: MutableMapping[str, float] = {}

    def record_event(
        self,
        kind: str,
        success: bool,
        duration: float,
        attributes: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not self.config.enabled:
            return

        event = AlertEvent(
            timestamp=time.time(),
            kind=kind,
            success=success,
            duration=duration,
            attributes=dict(attributes or {}),
        )

        with self._lock:
            self._events.append(event)
            self._trim_events_locked()
            if kind == "http":
                self._evaluate_http_locked()

    def _trim_events_locked(self) -> None:
        cutoff = time.time() - self.config.evaluation_window_seconds
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

    def _evaluate_http_locked(self) -> None:
        http_events = [event for event in self._events if event.kind == "http"]
        total = len(http_events)
        if total < self.config.minimum_requests:
            return

        errors = sum(1 for event in http_events if not event.success)
        error_rate = errors / total if total else 0.0
        if error_rate >= self.config.error_rate_threshold:
            self._emit_alert_locked(
                name="http_error_rate",
                severity="critical" if error_rate >= self.config.error_rate_threshold * 2 else "warning",
                summary=f"HTTP error rate {error_rate:.2%} exceeds threshold {self.config.error_rate_threshold:.2%}",
                details={
                    "total": total,
                    "errors": errors,
                    "error_rate": error_rate,
                },
            )

        slow_events = [event for event in http_events if event.duration >= self.config.latency_threshold_seconds]
        if slow_events:
            worst = max(slow_events, key=lambda event: event.duration)
            severity = "critical" if worst.duration >= self.config.latency_threshold_seconds * 2 else "warning"
            self._emit_alert_locked(
                name="http_latency",
                severity=severity,
                summary=(
                    f"HTTP latency {worst.duration:.3f}s exceeded threshold "
                    f"{self.config.latency_threshold_seconds:.3f}s"
                ),
                details={
                    "duration": worst.duration,
                    "path": worst.attributes.get("route", "unknown"),
                    "method": worst.attributes.get("method", "unknown"),
                },
            )

    def _emit_alert_locked(
        self,
        *,
        name: str,
        severity: str,
        summary: str,
        details: Mapping[str, object],
    ) -> None:
        now = time.time()
        last = self._last_emitted.get(name)
        if last is not None and now - last < self.config.cooldown_seconds:
            return
        self._last_emitted[name] = now

        message = AlertMessage(
            name=name,
            severity=severity,
            summary=summary,
            details=details,
        )
        for notifier in self._notifiers:
            try:
                notifier.notify(message)
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("Error when emitting alert %s via %s", name, notifier.__class__.__name__)


def create_alert_manager(
    config: AlertingConfig,
    notifiers: Optional[Sequence[BaseNotifier]] = None,
) -> BaseAlertManager:
    """Create an alert manager instance for the given configuration."""

    if not config.enabled:
        return NullAlertManager()
    return AlertManager(config=config, notifiers=notifiers)


__all__ = [
    "AlertManager",
    "AlertMessage",
    "BaseAlertManager",
    "BaseNotifier",
    "LoggingNotifier",
    "NullAlertManager",
    "create_alert_manager",
]
