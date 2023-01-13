# ===- CMakeLists.txt - PyCDE sources --------------------------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//

include(AddMLIRPython)

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

################################################################################
# Declare native Python extension
################################################################################

declare_mlir_python_sources(PyCDEPythonExtension)

declare_mlir_python_extension(PyCDEPythonExtension.Core
  MODULE_NAME _pycde
  ADD_TO_PARENT PyCDEPythonExtension
  SOURCES
    PyCDEModule.cpp 
  EMBED_CAPI_LINK_LIBS

  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Build composite binaries
################################################################################
set(PYCDE_PYTHON_PACKAGE_DIR "${CIRCT_PYTHON_PACKAGES_DIR}/pycde/")

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(PyCDEPythonCAPI
  INSTALL_COMPONENT CIRCTBindingsPythonModules
  INSTALL_DESTINATION python_packages/pycde/mlir/_mlir_libs
  # NOTE: When the MLIR API is relocated under circt, this would change to
  # .../circt/_mlir_libs
  OUTPUT_DIRECTORY "${PYCDE_PYTHON_PACKAGE_DIR}/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    MLIRPythonSources.Core
    CIRCTBindingsPythonSources
    CIRCTBindingsPythonExtension
    PyCDEPythonExtension
)

add_mlir_python_modules(PyCDE_MLIRPythonModules
  ROOT_PREFIX "${PYCDE_PYTHON_PACKAGE_DIR}/mlir"
  INSTALL_PREFIX "python_packages/pycde/mlir"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    CIRCTBindingsPythonExtension
    PyCDEPythonExtension
  COMMON_CAPI_LINK_LIBS
    PyCDEPythonCAPI
)

add_mlir_python_modules(PyCDE_CIRCTPythonModules
  ROOT_PREFIX "${PYCDE_PYTHON_PACKAGE_DIR}/"
  INSTALL_PREFIX "python_packages/pycde/"
  DECLARED_SOURCES
    CIRCTBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    PyCDEPythonCAPI
)

add_mlir_python_modules(PyCDE
  ROOT_PREFIX "${PYCDE_PYTHON_PACKAGE_DIR}/"
  INSTALL_PREFIX "python_packages/"
  DECLARED_SOURCES
    PyCDESources
  COMMON_CAPI_LINK_LIBS
    PyCDEPythonCAPI
)

add_dependencies(PyCDE
  PyCDE_CIRCTPythonModules
  PyCDE_MLIRPythonModules)
