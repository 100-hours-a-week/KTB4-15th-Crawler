import unittest

from diagnostics.compare_clients import compare
from price_refresh.source.client import MOST_REVIEWED, RECOMMENDED


class DiagnosticTests(unittest.TestCase):
    def test_success_runs_exactly_three_requests_and_waits_after_each(self):
        calls, waits = [], []

        def probe(kind, sort, legacy, codes):
            calls.append((kind, sort))
            return {"ok": True}

        records = compare(None, {}, probe=probe, sleep=waits.append)
        self.assertEqual(calls, [("legacy", MOST_REVIEWED), ("new", MOST_REVIEWED),
                                 ("new", RECOMMENDED)])
        self.assertEqual(len(records), 3)
        self.assertEqual(waits, [1.0, 1.0, 1.0])

    def test_either_comparison_failure_prevents_third_request(self):
        for results in ((False, True), (True, False), (False, False)):
            calls, waits = [], []

            def probe(*args):
                calls.append(args[0])
                return {"ok": results[len(calls) - 1]}

            records = compare(None, {}, probe=probe, sleep=waits.append)
            self.assertEqual(len(records), 2)
            self.assertEqual(waits, [1.0, 1.0])

    def test_unexpected_failure_is_not_retried_and_still_waits(self):
        waits = []

        def probe(*args):
            raise RuntimeError("stop")

        with self.assertRaises(RuntimeError):
            compare(None, {}, probe=probe, sleep=waits.append)
        self.assertEqual(waits, [1.0])


if __name__ == "__main__":
    unittest.main()
