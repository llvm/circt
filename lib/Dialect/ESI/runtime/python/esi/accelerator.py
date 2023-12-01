#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict

from .types import BundlePort
from . import esiCppAccel as cpp


class AcceleratorConnection:
  """An ESI accelerator."""

  def __init__(self, platform: str, connection_str: str):
    self.cpp_accel = cpp.AcceleratorConnection(platform, connection_str)

  def manifest(self) -> cpp.Manifest:
    """Get and parse the accelerator manifest."""
    return cpp.Manifest(self.cpp_accel.sysinfo().json_manifest())

  def sysinfo(self) -> cpp.SysInfo:
    return self.cpp_accel.sysinfo()

  def build_accelerator(self) -> "Accelerator":
    return Accelerator(self.manifest().build_accelerator(self.cpp_accel))


class HWModule:

  def __init__(self, cpp_hwmodule: cpp.HWModule):
    self.cpp_hwmodule = cpp_hwmodule

  @property
  def children(self) -> Dict[cpp.AppID, "Instance"]:
    return {
        name: Instance(inst)
        for name, inst in self.cpp_hwmodule.children.items()
    }

  @property
  def ports(self) -> Dict[cpp.AppID, BundlePort]:
    return {
        name: BundlePort(port)
        for name, port in self.cpp_hwmodule.ports.items()
    }


class Instance(HWModule):

  def __init__(self, cpp_instance: cpp.Instance):
    super().__init__(cpp_instance)
    self.cpp_hwmodule = cpp_instance

  @property
  def id(self) -> cpp.AppID:
    return self.cpp_hwmodule.id


class Accelerator(HWModule):
  """Root of the accelerator design hierarchy."""

  def __init__(self, cpp_accelerator: cpp.Accelerator):
    super().__init__(cpp_accelerator)
    self.cpp_hwmodule = cpp_accelerator
