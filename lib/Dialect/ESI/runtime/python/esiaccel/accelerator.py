# ===-----------------------------------------------------------------------===#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ===-----------------------------------------------------------------------===#
#
# The structure of the Python classes and hierarchy roughly mirrors the C++
# side, but wraps the C++ objects. The wrapper classes sometimes add convenience
# functionality and serve to return wrapped versions of the returned objects.
#
# ===-----------------------------------------------------------------------===#

from typing import Dict, List, Optional

from .types import BundlePort
from . import esiCppAccel as cpp

# Global context for the C++ side.
ctxt = cpp.Context()


class AcceleratorConnection:
  """A connection to an ESI accelerator."""

  def __init__(self, platform: str, connection_str: str):
    self.cpp_accel = cpp.AcceleratorConnection(ctxt, platform, connection_str)

  def manifest(self) -> cpp.Manifest:
    """Get and parse the accelerator manifest."""
    return cpp.Manifest(ctxt, self.cpp_accel.sysinfo().json_manifest())

  def sysinfo(self) -> cpp.SysInfo:
    return self.cpp_accel.sysinfo()

  def build_accelerator(self) -> "Accelerator":
    return Accelerator(self.manifest().build_accelerator(self.cpp_accel))

  def get_service_mmio(self) -> cpp.MMIO:
    return self.cpp_accel.get_service_mmio()

  def get_service_hostmem(self) -> cpp.HostMem:
    return self.cpp_accel.get_service_hostmem()


from .esiCppAccel import HostMemOptions


class HWModule:
  """Represents either the top level or an instance of a hardware module."""

  def __init__(self, parent: Optional["HWModule"], cpp_hwmodule: cpp.HWModule):
    self.parent = parent
    self.cpp_hwmodule = cpp_hwmodule

  @property
  def children(self) -> Dict[cpp.AppID, "Instance"]:
    return {
        name: Instance(self, inst)
        for name, inst in self.cpp_hwmodule.children.items()
    }

  @property
  def ports(self) -> Dict[cpp.AppID, BundlePort]:
    return {
        name: BundlePort(self, port)
        for name, port in self.cpp_hwmodule.ports.items()
    }

  @property
  def services(self) -> List[cpp.AppID]:
    return self.cpp_hwmodule.services


MMIO = cpp.MMIO


class Instance(HWModule):
  """Subclass of `HWModule` which represents a submodule instance. Adds an
  AppID, which the top level doesn't have or need."""

  def __init__(self, parent: Optional["HWModule"], cpp_instance: cpp.Instance):
    super().__init__(parent, cpp_instance)
    self.cpp_hwmodule: cpp.Instance = cpp_instance

  @property
  def id(self) -> cpp.AppID:
    return self.cpp_hwmodule.id


class Accelerator(HWModule):
  """Root of the accelerator design hierarchy."""

  def __init__(self, cpp_accelerator: cpp.Accelerator):
    super().__init__(None, cpp_accelerator)
    self.cpp_hwmodule = cpp_accelerator
