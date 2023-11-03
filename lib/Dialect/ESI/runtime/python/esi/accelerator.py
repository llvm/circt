#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import esiCppAccel as cpp

import json


class Accelerator(cpp.Accelerator):
  """A connection to an ESI accelerator."""

  @property
  def manifest(self) -> "AcceleratorManifest":
    """Get and parse the accelerator manifest."""
    return AcceleratorManifest(self)


class AcceleratorManifest(cpp.Manifest):
  """An accelerator manifest. Essential for interacting with an accelerator."""
  pass
