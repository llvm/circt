add_circt_unittest(CIRCTHWTests
  HWModuleTest.cpp
  InstanceGraphTest.cpp
  RLELogicTest.cpp
)

target_link_libraries(CIRCTHWTests
  PRIVATE
  CIRCTHW
)
