# ESI runtime requires exceptions to be enabled.
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)

add_circt_unittest(CIRCTESIRuntimeTests
    ESIRuntimeTest.cpp
)

target_link_libraries(CIRCTESIRuntimeTests PRIVATE
    esiaccel::ESICppRuntime
)
