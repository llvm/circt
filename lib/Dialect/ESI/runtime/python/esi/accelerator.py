#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .esiCppAccel import Accelerator as CppAccelerator

import json


class Accelerator(CppAccelerator):
  """A connection to an ESI accelerator."""

  @property
  def manifest(self) -> "AcceleratorManifest":
    """Get and parse the accelerator manifest."""
    return AcceleratorManifest(self)


class AcceleratorManifest:
  """An accelerator manifest. Essential for interacting with an accelerator."""

  def __init__(self, accel: Accelerator) -> None:
    self.accel = accel
    self._manifest = json.loads(accel.sysinfo().json_manifest())
