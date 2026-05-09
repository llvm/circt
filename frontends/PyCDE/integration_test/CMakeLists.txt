# ===- CMakeLists.txt - PyCDE integration tests cmake ---------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//


if(NOT ESI_RUNTIME)
  message(WARNING "PyCDE integration tests require ESI_RUNTIME to be enabled")
else()
  set(PYCDE_INTEGRATION_TEST_DEPENDS
    FileCheck count not split-file
    circt-rtl-sim
    PyCDE
    mlir-opt
    hlstool
    CIRCTPythonModules
    ESIRuntime
    ESIPythonRuntime
    esitester
  )

  # If ESI Cosim is available to build then enable its tests.
  if (TARGET EsiCosimDpiServer)
    list(APPEND PYCDE_INTEGRATION_TEST_DEPENDS
      EsiCosimDpiServer
      esi-cosim
    )

  endif()

  set(PYCDE_INTEGRATION_TIMEOUT 180) # Set a 180s timeout on individual tests.
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    PyCDE_INTEGRATION_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

  add_lit_testsuite(check-pycde-integration "Running the PyCDE integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${PYCDE_INTEGRATION_TEST_DEPENDS}
    )
  set_target_properties(check-pycde-integration PROPERTIES FOLDER "PyCDEIntegrationTests")
endif()
