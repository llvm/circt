add_circt_unittest(CIRCTHWTests
  InstanceGraphTest.cpp
)

target_link_libraries(CIRCTHWTests
  PRIVATE
  CIRCTHW
)
