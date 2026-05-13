#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pytest harness for the codegen + port-kind coverage tests.

Where ``test_serialization_probes`` exercises wire-format invariants, this
suite exercises the *port-kind* surface area of the ESI runtime + facade
codegen end-to-end. It builds the C++ driver under ``sw/test_codegen.cpp``
against generated ESI facade headers and runs each probe individually
against a cosim-driven instance of ``hw/test_codegen.py``.
"""

from __future__ import annotations

from pathlib import Path

from esiaccel.cosim.pytest import cosim_test

from .conftest import HW_DIR, build_cpp_test, run_probe


def _build(sources_dir: Path) -> Path:
  return build_cpp_test(sources_dir, "test_codegen_test", "test_codegen")


@cosim_test(HW_DIR / "test_codegen.py")
class TestCodegen:
  """One pytest method per probe; each gets its own cosim simulator process
  but shares the (cached) build of ``test_codegen_test``."""

  def test_typed_func_multi_arg(self, host: str, port: int,
                                sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "typed_func_multi_arg")

  def test_typed_func_void_arg(self, host: str, port: int,
                               sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "typed_func_void_arg")

  def test_typed_func_void_result(self, host: str, port: int,
                                  sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "typed_func_void_result")

  def test_call_service_callback(self, host: str, port: int,
                                 sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "call_service_callback")

  def test_typed_read_channel_struct(self, host: str, port: int,
                                     sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "typed_read_channel_struct")

  def test_typed_write_channel_byte(self, host: str, port: int,
                                    sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "typed_write_channel_byte")

  def test_mmio_read_write(self, host: str, port: int,
                           sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "mmio_read_write")

  def test_telemetry_metric(self, host: str, port: int,
                            sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "telemetry_metric")

  def test_indexed_func_group(self, host: str, port: int,
                              sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "indexed_func_group")

  def test_custom_service_decl_channel_0(self, host: str, port: int,
                                         sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "custom_service_decl_channel_0")

  def test_custom_service_decl_channel_1(self, host: str, port: int,
                                         sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "custom_service_decl_channel_1")

  def test_typed_func_struct(self, host: str, port: int,
                             sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "typed_func_struct")

  def test_typed_func_nested_struct(self, host: str, port: int,
                                    sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "typed_func_nested_struct")

  def test_typed_func_subbyte_signed(self, host: str, port: int,
                                     sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "typed_func_subbyte_signed")

  def test_typed_func_array_result(self, host: str, port: int,
                                   sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "typed_func_array_result")

  def test_typed_func_windowed_list(self, host: str, port: int,
                                    sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "typed_func_windowed_list")

  def test_channel_windowed_list_read(self, host: str, port: int,
                                      sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "channel_windowed_list_read")

  def test_channel_windowed_list_write(self, host: str, port: int,
                                       sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "channel_windowed_list_write")

  def test_callback_windowed_list(self, host: str, port: int,
                                  sources_dir: Path) -> None:
    run_probe(_build(sources_dir), host, port, "callback_windowed_list")
