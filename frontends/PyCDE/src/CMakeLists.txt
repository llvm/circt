# ===- CMakeLists.txt - PyCDE sources --------------------------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//

include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=pycde.circt.")

declare_mlir_python_sources(PyCDESources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
  __init__.py
  types.py
  dialects/__init__.py
  dialects/msft.py
  dialects/seq.py
  dialects/hw.py
  dialects/comb.py
  dialects/sv.py
  dialects/fsm.py
  dialects/esi.py
  dialects/hwarith.py
  support.py
  module.py
  behavioral.py
  constructs.py
  common.py
  system.py
  devicedb.py
  instance.py
  value.py
  ndarray.py
  esi.py
  esi_api.py
  esi_runtime_common.py
  fsm.py
  testing.py

  esi_api.py.j2
  Makefile.cosim
)

################################################################################
# Build composite binaries
################################################################################
set(PYCDE_PYTHON_PACKAGE_DIR "${CIRCT_PYTHON_PACKAGES_DIR}/pycde/")

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(PyCDE_CIRCTPythonCAPI
  INSTALL_COMPONENT CIRCTBindingsPythonModules
  INSTALL_DESTINATION python_packages/pycde/pycde/circt/_mlir_libs
  OUTPUT_DIRECTORY "${PYCDE_PYTHON_PACKAGE_DIR}/pycde/circt/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    MLIRPythonSources.Core
    CIRCTBindingsPythonExtension
)

add_mlir_python_modules(PyCDE_CIRCTPythonModules
  ROOT_PREFIX "${PYCDE_PYTHON_PACKAGE_DIR}/pycde/circt"
  INSTALL_PREFIX "python_packages/pycde/pycde/circt"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    CIRCTBindingsPythonExtension
    CIRCTBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    PyCDE_CIRCTPythonCAPI
)

add_mlir_python_modules(PyCDE
  ROOT_PREFIX "${PYCDE_PYTHON_PACKAGE_DIR}/pycde"
  INSTALL_PREFIX "python_packages/pycde/pycde"
  DECLARED_SOURCES
    PyCDESources
  COMMON_CAPI_LINK_LIBS
    PyCDE_CIRCTPythonCAPI
)

add_dependencies(PyCDE
  PyCDE_CIRCTPythonModules
)
