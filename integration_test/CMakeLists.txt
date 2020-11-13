set(CIRCT_INTEGRATION_TEST_DEPENDS
  FileCheck count not
  circt-opt
  circt-translate)

# If ESI Cosim is available to build then enable its tests.
if (TARGET EsiCosimDpiServer)
  list(APPEND CIRCT_INTEGRATION_TEST_DEPENDS EsiCosimDpiServer)
  get_property(ESI_COSIM_LIB_DIR TARGET EsiCosimDpiServer PROPERTY LIBRARY_OUTPUT_DIRECTORY)
  set(ESI_COSIM_PATH ${ESI_COSIM_LIB_DIR}/libEsiCosimDpiServer.so)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  INTEGRATION_CONFIG
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
