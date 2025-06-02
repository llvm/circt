#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .accelerator import AcceleratorConnection

from .esiCppAccel import (AppID, Type, BundleType, ChannelType, ArrayType,
                          StructType, BitsType, UIntType, SIntType)

__all__ = [
    "AcceleratorConnection", "AppID", "Type", "BundleType", "ChannelType",
    "ArrayType", "StructType", "BitsType", "UIntType", "SIntType"
]


def get_cmake_dir():
  """
    Returns the directory where the CMake files for the ESI runtime are located.
    """
  import os
  from pathlib import Path

  return os.path.join(os.path.dirname(__file__), "cmake")
