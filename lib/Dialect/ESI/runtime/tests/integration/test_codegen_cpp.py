#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pytest harness for the codegen + port-kind coverage tests.

Where ``test_serialization_probes`` exercises wire-format invariants, this
suite exercises the *port-kind* surface area of the ESI runtime + facade
codegen end-to-end. It builds the C++ driver under ``sw/test_codegen.cpp``
against generated ESI facade headers and runs each probe individually
against a cosim-driven instance of ``hw/test_codegen.py``.

Two test classes are generated — one for the default ``cosim`` BSP and one
for ``cosim_dma`` — so every probe is exercised through both channel
transport paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from esiaccel.cosim.pytest import cosim_test

from .conftest import HW_DIR, build_cpp_test, run_probe

PROBES = [
    "typed_func_multi_arg",
    "typed_func_void_arg",
    "typed_func_void_result",
    "call_service_callback",
    "typed_read_channel_struct",
    "typed_write_channel_byte",
    "mmio_read_write",
    "telemetry_metric",
    "indexed_func_group",
    "custom_service_decl_channel_0",
    "custom_service_decl_channel_1",
    "typed_func_struct",
    "typed_func_nested_struct",
    "typed_func_subbyte_signed",
    "typed_func_array_result",
    "typed_func_windowed_list",
    "channel_windowed_list_read",
    "channel_multiburst_list_read",
    "channel_windowed_list_write",
    "channel_multiburst_list_write",
    "callback_windowed_list",
]


def _build(sources_dir: Path) -> Path:
  return build_cpp_test(sources_dir, "test_codegen_test", "test_codegen")


def _make_probe_test(probe: str):
  """Create a test method that runs a single probe."""

  def test_method(self, host: str, port: int, sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, probe)

  test_method.__name__ = f"test_{probe}"
  test_method.__qualname__ = f"<dynamic>.test_{probe}"
  return test_method


def _make_codegen_class(name: str, bsp_args):
  """Build a test class with one method per probe, decorated with cosim_test."""
  ns = {f"test_{p}": _make_probe_test(p) for p in PROBES}
  cls = type(name, (), ns)
  return cosim_test(HW_DIR / "test_codegen.py", args=bsp_args)(cls)


TestCodegen = _make_codegen_class("TestCodegen", ("{tmp_dir}",))
TestCodegenDma = _make_codegen_class("TestCodegenDma",
                                     ("{tmp_dir}", "cosim_dma"))
