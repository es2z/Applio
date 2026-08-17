"""Stable PortAudio device registry.

PortAudio indices are runtime identifiers and may move after a refresh or reboot.
The UI therefore stores a deterministic fingerprint and resolves it against one
metadata snapshot when Start is pressed.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass

import sounddevice as sd


Direction = str


def _normalise(value: str) -> str:
    return " ".join(str(value).casefold().split())


@dataclass(frozen=True)
class AudioDeviceRef:
    index: int
    name: str
    host_api: str
    host_api_index: int
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    direction: Direction
    duplicate_ordinal: int
    fingerprint: str

    @property
    def label(self) -> str:
        return f"PA {self.index}: {self.name} ({self.host_api})"

    @property
    def channels(self) -> int:
        return (
            self.max_input_channels
            if self.direction == "input"
            else self.max_output_channels
        )


class AudioDeviceRegistry:
    def __init__(self, inputs: list[AudioDeviceRef], outputs: list[AudioDeviceRef]):
        self.inputs = inputs
        self.outputs = outputs
        self._by_token = {
            device.fingerprint: device for device in [*inputs, *outputs]
        }

    def resolve(self, token: str | None, direction: Direction) -> AudioDeviceRef:
        if not token:
            raise ValueError(f"No {direction} audio device was selected")
        token = str(token)
        device = self._by_token.get(token)
        if device is None:
            migrated = self.migrate_saved_value(token, direction)
            device = self._by_token.get(migrated) if migrated else None
        if device is None or device.direction != direction:
            raise ValueError(
                f"The selected {direction} device is no longer present. Refresh devices."
            )
        return device

    def choices(self, direction: Direction) -> list[tuple[str, str]]:
        devices = self.inputs if direction == "input" else self.outputs

        def priority(device: AudioDeviceRef) -> tuple[int, str, int]:
            name = device.name.casefold()
            virtual = 0 if "virtual" in name else 1 if "vb" in name else 2
            if direction == "input":
                virtual = -virtual
            return virtual, name, device.index

        return [
            (device.label, device.fingerprint)
            for device in sorted(devices, key=priority)
        ]

    def migrate_saved_value(
        self, saved: str | None, direction: Direction
    ) -> str | None:
        if not saved:
            return None
        saved = str(saved)
        direct = self._by_token.get(saved)
        if direct and direct.direction == direction:
            return direct.fingerprint

        # Legacy values look like "15: Name (Windows WDM-KS)". The leading
        # number was a sorted display ordinal, not a PortAudio id.
        portaudio_match = re.match(r"^\s*PA\s+(\d+)\s*:\s*", saved, re.I)
        requested_portaudio_index = (
            int(portaudio_match.group(1)) if portaudio_match else None
        )
        without_prefix = re.sub(r"^\s*(?:PA\s+)?\d+\s*:\s*", "", saved)
        match = re.match(r"^(.*)\s+\(([^()]*)\)\s*$", without_prefix)
        if match:
            wanted_name, wanted_host = map(_normalise, match.groups())
            devices = self.inputs if direction == "input" else self.outputs
            candidates = [
                device
                for device in devices
                if _normalise(device.name) == wanted_name
                and _normalise(device.host_api) == wanted_host
            ]
            if requested_portaudio_index is not None:
                exact = [
                    device
                    for device in candidates
                    if device.index == requested_portaudio_index
                ]
                if exact:
                    return exact[0].fingerprint
            if candidates:
                return candidates[0].fingerprint
        return None


_registry_lock = threading.Lock()
_registry: AudioDeviceRegistry | None = None


def _fingerprint(
    *,
    name: str,
    host_api: str,
    direction: Direction,
    max_input_channels: int,
    max_output_channels: int,
    duplicate_ordinal: int,
) -> str:
    raw = "|".join(
        (
            _normalise(host_api),
            direction,
            _normalise(name),
            str(max_input_channels),
            str(max_output_channels),
            str(duplicate_ordinal),
        )
    )
    return f"pa-{direction}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def refresh_audio_device_registry() -> AudioDeviceRegistry:
    """Query metadata only; do not open every advertised device."""
    global _registry
    raw_devices = list(sd.query_devices())
    hostapis = list(sd.query_hostapis())
    duplicate_counts: dict[tuple[str, str, Direction], int] = {}
    inputs: list[AudioDeviceRef] = []
    outputs: list[AudioDeviceRef] = []

    for index, raw in enumerate(raw_devices):
        host_api_index = int(raw["hostapi"])
        host_api = str(hostapis[host_api_index]["name"])
        name = str(raw["name"])
        for direction, channel_key, destination in (
            ("input", "max_input_channels", inputs),
            ("output", "max_output_channels", outputs),
        ):
            if int(raw[channel_key]) <= 0:
                continue
            key = (_normalise(host_api), _normalise(name), direction)
            ordinal = duplicate_counts.get(key, 0)
            duplicate_counts[key] = ordinal + 1
            destination.append(
                AudioDeviceRef(
                    index=int(raw.get("index", index)),
                    name=name,
                    host_api=host_api,
                    host_api_index=host_api_index,
                    max_input_channels=int(raw["max_input_channels"]),
                    max_output_channels=int(raw["max_output_channels"]),
                    default_samplerate=float(raw["default_samplerate"]),
                    direction=direction,
                    duplicate_ordinal=ordinal,
                    fingerprint=_fingerprint(
                        name=name,
                        host_api=host_api,
                        direction=direction,
                        max_input_channels=int(raw["max_input_channels"]),
                        max_output_channels=int(raw["max_output_channels"]),
                        duplicate_ordinal=ordinal,
                    ),
                )
            )

    with _registry_lock:
        _registry = AudioDeviceRegistry(inputs, outputs)
        return _registry


def get_audio_device_registry(*, refresh: bool = False) -> AudioDeviceRegistry:
    global _registry
    with _registry_lock:
        existing = _registry
    if refresh or existing is None:
        return refresh_audio_device_registry()
    return existing


# Backward-compatible metadata API for code outside the Realtime tab.
def list_audio_device():
    registry = get_audio_device_registry(refresh=True)
    return registry.inputs, registry.outputs
