add_subdirectory(CAPI)
add_subdirectory(Dialect)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(CIRCT_TEST_DEPENDS
  FileCheck count not
  circt-capi-ir-test
  circt-opt
  circt-translate
  esi-tester
  handshake-runner
  firtool
  mlir-opt
  mlir-cpu-runner
  )

if(CIRCT_LLHD_SIM_ENABLED)
  list(APPEND CIRCT_TEST_DEPENDS llhd-sim)
endif()

add_lit_testsuite(check-circt "Running the CIRCT regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_TEST_DEPENDS}
  )
set_target_properties(check-circt PROPERTIES FOLDER "Tests")

add_lit_testsuites(CIRCT ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CIRCT_TEST_DEPS}
)
