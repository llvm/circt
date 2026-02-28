llvm_canonicalize_cmake_booleans(
  CIRCT_BINDINGS_PYTHON_ENABLED
  )

set(CIRCT_INTEGRATION_TEST_DEPENDS
  FileCheck count not split-file
  arcilator
  circt-opt
  circt-translate
  circt-rtl-sim
  circt-lec
  circt-bmc
  circt-synth
  circt-test
  circt-test-runners
  firtool
  hlstool
  kanagawatool
  handshake-runner
  )

if (MLIR_ENABLE_EXECUTION_ENGINE)
  list(APPEND CIRCT_INTEGRATION_TEST_DEPENDS mlir-runner)
endif()

# If Python bindings are available to build then enable the tests.
if(CIRCT_BINDINGS_PYTHON_ENABLED)
  list(APPEND CIRCT_INTEGRATION_TEST_DEPENDS CIRCTPythonModules)
endif()


# If the ESI runtime build and test is enabled, require those components.
llvm_canonicalize_cmake_booleans(ESI_RUNTIME)
if (ESI_RUNTIME)
  list(APPEND CIRCT_INTEGRATION_TEST_DEPENDS
    ESIRuntime
    esitester
  )

  # If ESI Cosim is available to build then enable its tests.
  if (TARGET EsiCosimDpiServer)
    list(APPEND CIRCT_INTEGRATION_TEST_DEPENDS
      EsiCosimDpiServer
      esi-cosim
    )

  endif()
endif()

get_target_property(ESI_PrimsDir ESIPrimitives SOURCE_DIR)
set(ESI_Prims "${ESI_PrimsDir}/ESIPrimitives.sv")

set(CIRCT_INTEGRATION_TIMEOUT 120) # Set a 120s timeout on individual tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-circt-integration "Running the CIRCT integration tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_INTEGRATION_TEST_DEPENDS}
  )
set_target_properties(check-circt-integration PROPERTIES FOLDER "IntegrationTests")

add_lit_testsuites(CIRCT_INTEGRATION ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CIRCT_INTEGRATION_TEST_DEPS}
)
