#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

def _load_extension(name):
  # TODO: Remote the RTLD_GLOBAL hack once local, cross module imports
  # resolve symbols properly. Something is keeping the dynamic loader on
  # Linux from treating the following vague symbols as the same across
  # _mlir and _circt:
  #   mlir::detail::TypeIDExported::get<mlir::FuncOp>()::instance
  import sys
  import ctypes
  import importlib
  flags = sys.getdlopenflags()
  sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)
  from mlir._cext_loader import _cext
  module = importlib.import_module(name)
  sys.setdlopenflags(flags)
  _cext.globals.append_dialect_search_prefix("circt.dialects")
  return module

def _reexport_cext(source_module, target_module_name):
  """Re-exports a named sub-module of the C-Extension into another module.

  Typically:
    from circt import _reexport_cext
    _reexport_cext("_dialects._rtl", __name__)
    del _reexport_cext
  """
  import sys
  target_module = sys.modules[target_module_name]
  for attr_name in dir(source_module):
    if not attr_name.startswith("__"):
      setattr(target_module, attr_name, getattr(source_module, attr_name))

_cext = _load_extension("_circt")
_reexport_cext(_cext, __name__)
