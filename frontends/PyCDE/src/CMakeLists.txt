# ===- CMakeLists.txt - PyCDE sources --------------------------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//

declare_mlir_python_sources(PyCDESources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES_GLOB pycde/*.py)

add_mlir_python_modules(PyCDE
  ROOT_PREFIX "${CIRCT_PYTHON_PACKAGES_DIR}/pycde"
  INSTALL_PREFIX "python_packages/pycde"
  DECLARED_SOURCES
    PyCDESources
)
