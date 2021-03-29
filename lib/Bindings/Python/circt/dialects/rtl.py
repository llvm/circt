#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Generated tablegen dialects end up in the mlir.dialects package for now.
from mlir.dialects._rtl_ops_gen import *

from circt import _load_extension, _reexport_cext

# Re-export the pybind11 module.
_cext = _load_extension("_circt._dialects._rtl")
_reexport_cext(_cext, __name__)

del _reexport_cext
