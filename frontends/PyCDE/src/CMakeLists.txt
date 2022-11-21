# ===- CMakeLists.txt - PyCDE sources --------------------------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//

declare_mlir_python_sources(PyCDESources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
  pycde/__init__.py
  pycde/pycde_types.py
  pycde/dialects/__init__.py
  pycde/dialects/msft.py
  pycde/dialects/seq.py
  pycde/dialects/hw.py
  pycde/dialects/comb.py
  pycde/dialects/sv.py
  pycde/dialects/fsm.py
  pycde/dialects/hwarith.py
  pycde/support.py
  pycde/module.py
  pycde/behavioral.py
  pycde/constructs.py
  pycde/common.py
  pycde/system.py
  pycde/devicedb.py
  pycde/instance.py
  pycde/value.py
  pycde/ndarray.py
  pycde/esi.py
  pycde/esi_api.py
  pycde/esi_runtime_common.py
  pycde/fsm.py
  pycde/testing.py

  pycde/esi_api.py.j2
  pycde/Makefile.cosim
)

add_mlir_python_modules(PyCDE
  ROOT_PREFIX "${CIRCT_PYTHON_PACKAGES_DIR}/pycde"
  INSTALL_PREFIX "python_packages/pycde"
  DECLARED_SOURCES
  PyCDESources
)
