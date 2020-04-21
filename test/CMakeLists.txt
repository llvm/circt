configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(SPT_TEST_DEPENDS
  FileCheck count not
  spt-opt
  spt-translate
  firtool
  )

add_lit_testsuite(check-spt "Running the SPT regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SPT_TEST_DEPENDS}
  )
set_target_properties(check-spt PROPERTIES FOLDER "Tests")

add_lit_testsuites(SPT ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${SPT_TEST_DEPS}
)
