include_directories(${CIRCT_MAIN_SRC_DIR})
add_circt_unittest(CIRCTLLHDTests DNFTest.cpp)
target_link_libraries(CIRCTLLHDTests PRIVATE CIRCTLLHDTransforms)
