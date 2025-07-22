add_subdirectory(lib)
add_subdirectory(CAPI)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

set(CIRCT_TEST_DEPENDS
  FileCheck count not
  split-file
  arcilator
  circt-capi-aig-test
  circt-capi-ir-test
  circt-capi-om-test
  circt-capi-firrtl-test
  circt-capi-firtool-test
  circt-capi-rtg-pipelines-test
  circt-capi-rtg-test
  circt-capi-rtgtest-test
  circt-capi-support-test
  circt-as
  circt-bmc
  circt-dis
  circt-lec
  circt-opt
  circt-synth
  circt-test
  circt-translate
  circt-reduce
  handshake-runner
  firld
  firtool
  hlstool
  kanagawatool
  om-linker
  )

if (CIRCT_GTEST_AVAILABLE)
  list(APPEND CIRCT_TEST_DEPENDS CIRCTUnitTests)
endif()

if(CIRCT_SLANG_FRONTEND_ENABLED)
  list(APPEND CIRCT_TEST_DEPENDS circt-verilog)
  list(APPEND CIRCT_TEST_DEPENDS circt-verilog-lsp-server)
endif()

add_lit_testsuite(check-circt "Running the CIRCT regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_TEST_DEPENDS}
  )
set_target_properties(check-circt PROPERTIES FOLDER "Tests")

add_lit_testsuites(CIRCT ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_TEST_DEPENDS}
  )
