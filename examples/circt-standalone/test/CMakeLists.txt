llvm_canonicalize_cmake_booleans(
  CIRCT_STANDALONE_HAS_PLUGIN
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CIRCT_STANDALONE_TEST_DEPENDS
  FileCheck
  circt-standalone-opt
)
if(TARGET CIRCTStandalonePlugin)
  list(APPEND CIRCT_STANDALONE_TEST_DEPENDS CIRCTStandalonePlugin)
endif()

add_lit_testsuite(check-circt-standalone
  "Running the CIRCT standalone regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_STANDALONE_TEST_DEPENDS}
)
set_target_properties(check-circt-standalone PROPERTIES FOLDER "Tests")

add_lit_testsuites(CIRCT_STANDALONE ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CIRCT_STANDALONE_TEST_DEPENDS}
)
