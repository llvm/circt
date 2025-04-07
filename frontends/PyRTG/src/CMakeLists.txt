# ===- CMakeLists.txt - PyRTG sources -------------------------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=pyrtg.circt.")

declare_mlir_python_sources(PyRTGSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pyrtg"
  SOURCES
    __init__.py
    base.py
    contexts.py
    resources.py
    rtgtest.py
    support.py
)

declare_mlir_python_sources(PyRTGSources.Core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pyrtg"
  ADD_TO_PARENT PyRTGSources
  SOURCES
    arrays.py
    bags.py
    configs.py
    control_flow.py
    core.py
    index.py
    integers.py
    labels.py
    memories.py
    rtg.py
    scf.py
    sequences.py
    sets.py
    tests.py
    tuples.py
)

declare_mlir_python_sources(PyRTGToolSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rtgtool"
  SOURCES
    rtgtool.py
)

################################################################################
# Build composite binaries
################################################################################

set(PYRTG_PYTHON_PACKAGE_DIR "${CIRCT_PYTHON_PACKAGES_DIR}/pyrtg/")

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(PyRTG_CIRCTPythonCAPI
  INSTALL_COMPONENT CIRCTBindingsPythonModules
  INSTALL_DESTINATION python_packages/pyrtg/circt/_mlir_libs
  OUTPUT_DIRECTORY "${PYRTG_PYTHON_PACKAGE_DIR}/pyrtg/circt/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    MLIRPythonSources.Core
    CIRCTBindingsPythonExtension
)

add_mlir_python_modules(PyRTG_CIRCTPythonModules
  ROOT_PREFIX "${PYRTG_PYTHON_PACKAGE_DIR}/pyrtg/circt"
  INSTALL_PREFIX "python_packages/pyrtg/circt"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    CIRCTBindingsPythonExtension
    CIRCTBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    PyRTG_CIRCTPythonCAPI
)

add_mlir_python_modules(PyRTG
  ROOT_PREFIX "${PYRTG_PYTHON_PACKAGE_DIR}/pyrtg"
  INSTALL_PREFIX "python_packages/pyrtg"
  DECLARED_SOURCES
    PyRTGSources
  COMMON_CAPI_LINK_LIBS
    PyRTG_CIRCTPythonCAPI
)

add_mlir_python_modules(PyRTGTool
  ROOT_PREFIX "${PYRTG_PYTHON_PACKAGE_DIR}/rtgtool"
  INSTALL_PREFIX "python_packages/pyrtg/rtgtool"
  DECLARED_SOURCES
    PyRTGToolSources
)

install(TARGETS PyRTG_CIRCTPythonCAPI
  DESTINATION python_packages/pyrtg/circt/_mlir_libs
  RUNTIME_DEPENDENCIES
    PRE_EXCLUDE_REGEXES ".*"
    PRE_INCLUDE_REGEXES ".*zlib.*"
  COMPONENT PyRTG
)

add_dependencies(PyRTGTool PyRTG)
add_dependencies(install-PyRTGTool install-PyRTG)

add_dependencies(PyRTG PyRTG_CIRCTPythonModules)
add_dependencies(install-PyRTG install-PyRTG_CIRCTPythonModules)
