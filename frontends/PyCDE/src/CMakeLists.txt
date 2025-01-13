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
  pycde/__init__.py
  pycde/types.py
  pycde/dialects/__init__.py
  pycde/dialects/msft.py
  pycde/dialects/seq.py
  pycde/dialects/hw.py
  pycde/dialects/comb.py
  pycde/dialects/sv.py
  pycde/dialects/fsm.py
  pycde/dialects/esi.py
  pycde/dialects/hwarith.py
  pycde/support.py
  pycde/module.py
  pycde/behavioral.py
  pycde/constructs.py
  pycde/common.py
  pycde/system.py
  pycde/devicedb.py
  pycde/handshake.py
  pycde/instance.py
  pycde/seq.py
  pycde/signals.py
  pycde/ndarray.py
  pycde/esi.py
  pycde/fsm.py
  pycde/testing.py

  pycde/bsp/__init__.py
  pycde/bsp/common.py
  pycde/bsp/cosim.py
  pycde/bsp/xrt.py
  pycde/bsp/Makefile.xrt.mk
  pycde/bsp/xrt_package.tcl
  pycde/bsp/xrt.ini
  pycde/bsp/xsim.tcl
)

################################################################################
# Build composite binaries
################################################################################
set(PYCDE_PYTHON_PACKAGE_DIR "${CIRCT_PYTHON_PACKAGES_DIR}/pycde/")

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(PyCDE_CIRCTPythonCAPI
  INSTALL_COMPONENT CIRCTBindingsPythonModules
  INSTALL_DESTINATION python_packages/pycde/circt/_mlir_libs
  OUTPUT_DIRECTORY "${PYCDE_PYTHON_PACKAGE_DIR}/pycde/circt/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    MLIRPythonSources.Core
    CIRCTBindingsPythonExtension
)

add_mlir_python_modules(PyCDE_CIRCTPythonModules
  ROOT_PREFIX "${PYCDE_PYTHON_PACKAGE_DIR}/pycde/circt"
  INSTALL_PREFIX "python_packages/pycde/circt"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    CIRCTBindingsPythonExtension
    CIRCTBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    PyCDE_CIRCTPythonCAPI
)

add_mlir_python_modules(PyCDE
  ROOT_PREFIX "${PYCDE_PYTHON_PACKAGE_DIR}/"
  INSTALL_PREFIX "python_packages/"
  DECLARED_SOURCES
    PyCDESources
  COMMON_CAPI_LINK_LIBS
    PyCDE_CIRCTPythonCAPI
)

install(TARGETS PyCDE_CIRCTPythonCAPI
  DESTINATION python_packages/pycde/circt/_mlir_libs
  RUNTIME_DEPENDENCIES
    PRE_EXCLUDE_REGEXES ".*"
    PRE_INCLUDE_REGEXES ".*zlib.*"
  COMPONENT PyCDE
)
add_dependencies(PyCDE PyCDE_CIRCTPythonModules)
add_dependencies(install-PyCDE install-PyCDE_CIRCTPythonModules)

# Copy ESIPrimitives.sv to both the build and install directories.
# TODO: this won't work if ESIPrimitives has multiple source files. Figure out
# how to handle this.
set(esiprims "$<TARGET_PROPERTY:ESIPrimitives,SOURCE_DIR>/$<TARGET_PROPERTY:ESIPrimitives,SOURCES>")
add_custom_command(TARGET PyCDE POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${esiprims}
          ${PYCDE_PYTHON_PACKAGE_DIR}/pycde
)
install(FILES ${esiprims}
  DESTINATION python_packages/pycde
  COMPONENT PyCDE
)
