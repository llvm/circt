configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(CIRT_TEST_DEPENDS
  FileCheck count not
  cirt-opt
  cirt-translate
  firtool
  )

add_lit_testsuite(check-cirt "Running the CIRT regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRT_TEST_DEPENDS}
  )
set_target_properties(check-cirt PROPERTIES FOLDER "Tests")

add_lit_testsuites(CIRT ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CIRT_TEST_DEPS}
)
