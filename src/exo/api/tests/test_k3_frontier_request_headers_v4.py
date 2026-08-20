"""Focused wire-admission tests for sanitized frontier request headers."""

from __future__ import annotations

import hashlib
import unittest

from fastapi import HTTPException
from starlette.requests import Request

from exo.api.main import _k3_frontier_request_controls


def _request(headers: list[tuple[bytes, bytes]]) -> Request:
    return Request({"type": "http", "headers": headers})


class TestK3FrontierRequestHeadersV4(unittest.TestCase):
    def test_absent_headers_preserve_default_off_fields(self) -> None:
        self.assertEqual(
            _k3_frontier_request_controls(_request([])),
            {
                "k3_frontier_request_index": None,
                "k3_frontier_request_nonce_sha256": None,
            },
        )

    def test_exact_pair_is_admitted_without_request_body_mutation(self) -> None:
        nonce = hashlib.sha256(b"request-1").hexdigest()
        scope = _request(
            [
                (b"x-exo-k3-frontier-request-index", b"1"),
                (b"x-exo-k3-frontier-request-nonce-sha256", nonce.encode()),
            ]
        )
        self.assertEqual(
            _k3_frontier_request_controls(scope),
            {
                "k3_frontier_request_index": 1,
                "k3_frontier_request_nonce_sha256": nonce,
            },
        )

    def test_missing_duplicate_or_noncanonical_headers_fail(self) -> None:
        nonce = hashlib.sha256(b"request-1").hexdigest().encode()
        cases = [
            [(b"x-exo-k3-frontier-request-index", b"1")],
            [
                (b"x-exo-k3-frontier-request-index", b"1"),
                (b"x-exo-k3-frontier-request-index", b"1"),
                (b"x-exo-k3-frontier-request-nonce-sha256", nonce),
            ],
            [
                (b"x-exo-k3-frontier-request-index", b"01"),
                (b"x-exo-k3-frontier-request-nonce-sha256", nonce),
            ],
            [
                (b"x-exo-k3-frontier-request-index", b"17"),
                (b"x-exo-k3-frontier-request-nonce-sha256", nonce),
            ],
            [
                (b"x-exo-k3-frontier-request-index", b"1"),
                (b"x-exo-k3-frontier-request-nonce-sha256", b"A" * 64),
            ],
        ]
        for headers in cases:
            with self.subTest(headers=headers), self.assertRaises(HTTPException):
                _k3_frontier_request_controls(_request(headers))


if __name__ == "__main__":
    unittest.main()
