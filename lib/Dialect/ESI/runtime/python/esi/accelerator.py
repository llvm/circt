#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import esiCppAccel as cpp


class AcceleratorConnection(cpp.AcceleratorConnection):
  """An ESI accelerator."""

  def manifest(self) -> cpp.Manifest:
    """Returns the accelerator's manifest."""
    return cpp.Manifest(self.sysinfo().json_manifest())
