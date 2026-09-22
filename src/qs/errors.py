"""The station's exception hierarchy — every reason the Pipeline can be told about."""

from __future__ import annotations


class QSError(Exception):
    """Base class for every error the station raises."""


class ConfigError(QSError):
    """The station config is invalid: the station refuses to start (exit code 2)."""


class CaptureError(QSError):
    """No frame after the trigger within the timeout, or a corrupt frame."""

    reason: str = "capture"


class CameraLost(CaptureError):
    """The camera is gone: unplugged, or its device error is fatal."""

    reason: str = "camera"


class CameraSettingsDrift(QSError):
    """A readable camera setting no longer matches the config."""

    reason: str = "camera_settings"


class LightingFault(QSError):
    """The reference patch read outside its configured band."""

    reason: str = "lighting"


class PlacementFault(QSError):
    """No unit, more than one unit, or the unit is not wholly in the inspection area."""

    reason: str = "placement"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class InferenceError(QSError):
    """The model failed: CUDA error, non-finite score, any exception inside inspect()."""

    reason: str = "inference"
