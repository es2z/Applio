import unittest
from unittest.mock import Mock, patch

from rvc.realtime.compile_session import RvcCompileSession
from tabs.settings.sections.torch_compile import RealtimeCompileSettings


class RvcCompileSessionTests(unittest.TestCase):
    def test_live_failure_falls_back_without_global_cache_reset(self):
        eager = Mock(return_value="eager result")
        compiled = Mock(side_effect=RuntimeError("compiled graph failed"))
        session = RvcCompileSession.__new__(RvcCompileSession)
        session.eager = eager
        session.callable = compiled
        session.settings = RealtimeCompileSettings(rvc_enabled=True)
        session.signature = "test"
        session.namespace = Mock()
        session.status = Mock(active=True, backend="inductor/default", error=None)
        session._repair_allowed = False

        with patch(
            "rvc.realtime.compile_session.reset_failed_compile_namespace"
        ) as reset:
            self.assertEqual(session("input"), "eager result")

        reset.assert_not_called()
        eager.assert_called_once_with("input")
        self.assertFalse(session.status.active)
        self.assertEqual(session.status.backend, "eager-fallback")


if __name__ == "__main__":
    unittest.main()
