#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .esiCppAccel import Accelerator as CppAccelerator

import json


class Accelerator(CppAccelerator):

  @property
  def manifest(self) -> "AcceleratorManifest":
    return AcceleratorManifest(self)


class AcceleratorManifest:

  def __init__(self, accel) -> None:
    self.accel = accel
    self._manifest = json.loads(accel.sysinfo().json_manifest())
