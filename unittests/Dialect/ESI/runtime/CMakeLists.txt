# ESI runtime requires exceptions to be enabled.
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_circt_unittest(CIRCTESIRuntimeTests
    ESIRuntimeTest.cpp
    ESIRuntimeValuesTest.cpp
)

target_link_libraries(CIRCTESIRuntimeTests PRIVATE
    esiaccel::ESICppRuntime
)
