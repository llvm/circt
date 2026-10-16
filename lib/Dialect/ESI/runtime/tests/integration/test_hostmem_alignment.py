#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import sys

import esiaccel
import pytest
from esiaccel.accelerator import AcceleratorConnection, HostMemOptions
from esiaccel.cosim.pytest import cosim_test

from .conftest import HW_DIR


def test_hostmem_options():
  opts = HostMemOptions()
  assert opts.alignment == 0
  assert not opts.writeable
  assert not opts.use_large_pages
  assert repr(opts) == "HostMemOptions()"
  opts.alignment = 32
  assert repr(opts) == "HostMemOptions(alignment=32)"
  opts.writeable = True
  opts.use_large_pages = True
  assert repr(opts) == "HostMemOptions(writeable use_large_pages alignment=32)"
  with pytest.raises(TypeError):
    opts.alignment = -1


def _check_hostmem(hostmem):
  for alignment in (0, 1, 2, 8, 16, 32, 64, 4096, 8192):
    for writeable, large_pages in ((False, False), (True, True)):
      opts = HostMemOptions()
      opts.alignment = alignment
      opts.writeable = writeable
      opts.use_large_pages = large_pages
      assert hostmem.allocate(0, opts) is None
      for size in (1, 7, 33, 4097):
        region = hostmem.allocate(size, opts)
        assert region is not None
        assert region.size == size
        if alignment:
          assert region.ptr % alignment == 0
          assert region.device_ptr % alignment == 0
        data = bytes(i % 256 for i in range(size))
        ctypes.memmove(region.ptr, data, size)
        assert ctypes.string_at(region.ptr, size) == data
        assert hostmem.map_memory(region.ptr, size, opts)
        hostmem.unmap_memory(region.ptr)
        if alignment > 1:
          assert not hostmem.map_memory(region.ptr + 1, size - 1, opts)

  opts = HostMemOptions()
  opts.alignment = 32
  region = hostmem.allocate(33, opts)
  assert region is not None
  for invalid in (3, 6, 24, 4095):
    opts.alignment = invalid
    assert hostmem.allocate(33, opts) is None
    assert not hostmem.map_memory(region.ptr, region.size, opts)
  opts.alignment = 32
  assert hostmem.allocate(ctypes.c_size_t(-1).value, opts) is None

  default_region = hostmem.allocate(33)
  assert default_region is not None
  assert default_region.size == 33
  assert hostmem.map_memory(default_region.ptr + 1, 32)
  hostmem.unmap_memory(default_region.ptr + 1)


def test_trace_hostmem_alignment(tmp_path):
  manifest = tmp_path / "manifest.json"
  manifest.write_text("{}")
  separator = ";" if sys.platform == "win32" else ":"
  ctxt = esiaccel.Context()
  conn = ctxt.connect("trace", f"-{separator}{manifest}")
  _check_hostmem(conn.get_service_hostmem())


@cosim_test(HW_DIR / "hostmem_read.py", args=("{tmp_dir}", "1"))
def test_cosim_hostmem_alignment(conn: AcceleratorConnection):
  hostmem = conn.get_service_hostmem()
  _check_hostmem(hostmem)
  opts = HostMemOptions()
  opts.alignment = 32
  region = hostmem.allocate(33, opts)
  assert region is not None
  assert ctypes.string_at(region.ptr, region.size) == b"\xff" * 33
