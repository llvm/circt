add_subdirectory(lib)
add_subdirectory(CAPI)

# Find the concrete Z3 shared library for circt-bmc JIT tests. FindZ3 exposes
# an absolute path, while a Z3 package configuration generally exposes an
# imported target whose location is configuration-specific.
set(CIRCT_Z3_SHARED_LIBRARY "")
if(Z3_FOUND)
  if(TARGET z3::libz3)
    get_target_property(z3_imported_location z3::libz3 IMPORTED_LOCATION)
    if(z3_imported_location)
      set(CIRCT_Z3_SHARED_LIBRARY "${z3_imported_location}")
    else()
      get_target_property(z3_imported_configurations z3::libz3
                          IMPORTED_CONFIGURATIONS)
      foreach(z3_configuration IN LISTS z3_imported_configurations)
        string(TOUPPER "${z3_configuration}" z3_configuration_upper)
        get_target_property(z3_imported_location z3::libz3
                            "IMPORTED_LOCATION_${z3_configuration_upper}")
        if(z3_imported_location)
          set(CIRCT_Z3_SHARED_LIBRARY "${z3_imported_location}")
          break()
        endif()
      endforeach()
    endif()
  elseif(Z3_LIBRARIES)
    list(GET Z3_LIBRARIES 0 CIRCT_Z3_SHARED_LIBRARY)
  endif()
endif()

if(MLIR_ENABLE_EXECUTION_ENGINE)
  set(CIRCT_BMC_JIT_ENABLED 1)
else()
  set(CIRCT_BMC_JIT_ENABLED 0)
endif()

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
  circt-capi-ir-test
  circt-capi-om-test
  circt-capi-firrtl-test
  circt-capi-firtool-test
  circt-capi-ltl-test
  circt-capi-rtg-test
  circt-capi-rtgtest-test
  circt-capi-support-test
  circt-capi-synth-test
  circt-as
  circt-bmc
  circt-dis
  circt-lec
  circt-opt
  circt-synth
  circt-test
  circt-translate
  circt-reduce
  circt-tblgen
  domaintool
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
