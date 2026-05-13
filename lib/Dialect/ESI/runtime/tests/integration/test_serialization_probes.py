#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pytest harness for the SerializationProbes integration test.

This builds the C++ driver under ``sw/serialization_probes.cpp`` against
generated ESI facade headers and runs it against a cosim-driven instance of
``hw/serialization_probes.py``. Each probe asserts an exact, position-revealing
result so any drift in the host serializer or deserializer (vs hardware)
fails loudly.
"""

from __future__ import annotations

from pathlib import Path

from esiaccel.cosim.pytest import cosim_test

from .conftest import HW_DIR, build_cpp_test, run_probe

PROBES = [
    "byte_rotate1",
    "byte_pattern_const",
    "byte_pattern_echo_eq",
    "sign_probe",
    "sign_probe13",
    "pack_probe",
    "bit_pack_probe",
    "array_probe",
]


@cosim_test(HW_DIR / "serialization_probes.py")
class TestSerializationProbes:
  """End-to-end serialization-correctness probes."""

  def test_serialization_probes_cpp(self, host: str, port: int,
                                    sources_dir: Path) -> None:
    binary = build_cpp_test(sources_dir, "serialization_probes_test",
                            "serialization_probes")
    for probe in PROBES:
      run_probe(binary, host, port, probe)
