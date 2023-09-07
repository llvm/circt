add_circt_unittest(CIRCTHWTests
  GraphFixture.cpp
  HWModuleTest.cpp
  InstanceGraphTest.cpp
  InstancePathTest.cpp
)

target_link_libraries(CIRCTHWTests
  PRIVATE
  CIRCTHW
)
