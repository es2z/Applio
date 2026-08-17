import unittest
from unittest.mock import patch

from rvc.realtime import devices


class DeviceRegistryTests(unittest.TestCase):
    @patch("rvc.realtime.devices.sd.query_hostapis")
    @patch("rvc.realtime.devices.sd.query_devices")
    def test_choices_use_real_portaudio_id_and_legacy_value_migrates(
        self, query_devices, query_hostapis
    ):
        query_hostapis.return_value = [{"name": "Windows WDM-KS"}]
        query_devices.return_value = [
            {
                "index": 37,
                "name": "Loopback (Loopback)",
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 44100,
            },
            {
                "index": 42,
                "name": "Speakers (VB-Audio Point)",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 8,
                "default_samplerate": 44100,
            },
        ]
        registry = devices.refresh_audio_device_registry()
        self.assertTrue(registry.choices("input")[0][0].startswith("PA 37:"))
        token = registry.migrate_saved_value(
            "15: Loopback (Loopback) (Windows WDM-KS)", "input"
        )
        self.assertEqual(registry.resolve(token, "input").index, 37)
        self.assertEqual(
            registry.resolve(
                "PA 37: Loopback (Loopback) (Windows WDM-KS)", "input"
            ).index,
            37,
        )

    def test_missing_token_is_explicit_error(self):
        registry = devices.AudioDeviceRegistry([], [])
        with self.assertRaisesRegex(ValueError, "no longer present"):
            registry.resolve("missing", "output")


if __name__ == "__main__":
    unittest.main()
