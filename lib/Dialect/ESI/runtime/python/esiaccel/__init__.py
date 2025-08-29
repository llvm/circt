#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import os
from .accelerator import AcceleratorConnection

from .esiCppAccel import (AppID, Type, BundleType, ChannelType, ArrayType,
                          StructType, BitsType, UIntType, SIntType)

__all__ = [
    "AcceleratorConnection", "AppID", "Type", "BundleType", "ChannelType",
    "ArrayType", "StructType", "BitsType", "UIntType", "SIntType"
]

if sys.platform == "win32":
  # Ensure that ESI libraries are in the dll path on Windows, if users
  # build against the esiaccel-provided prebuilt CMake/prebuilt libraries.
  from .utils import get_cmake_dir
  os.add_dll_directory(str(get_cmake_dir() / ".."))
