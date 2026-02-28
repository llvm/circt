add_circt_unittest(CIRCTESITests
  OpTest.cpp
)

target_link_libraries(CIRCTESITests
  PRIVATE
  CIRCTHW
  CIRCTESI
)

if(ESI_RUNTIME)
  add_subdirectory(runtime)
endif()
