add_circt_unittest(CIRCTFIRRTLFFITests
  FFIContext.cpp
)

target_link_libraries(CIRCTFIRRTLFFITests
  PRIVATE
  CIRCTFIRRTLFFI
)
