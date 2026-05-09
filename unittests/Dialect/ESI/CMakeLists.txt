add_circt_unittest(CIRCTESITests
  OpTest.cpp
)

target_link_libraries(CIRCTESITests
  PRIVATE
  CIRCTHW
  CIRCTESI
)

# ESI runtime C++ unit tests are built inside the runtime CMake project
# (lib/Dialect/ESI/runtime/tests/cpp) and run via pytest.
