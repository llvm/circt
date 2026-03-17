##===- CMakeLists.txt - ESI runtime C++ unit tests --------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
## Self-contained C++ unit tests for the ESI runtime. Built as part of the
## ESI runtime CMake project without any dependency on CIRCT or LLVM.
## GoogleTest is fetched automatically.
##
##===----------------------------------------------------------------------===//

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
  FIND_PACKAGE_ARGS NAMES GTest
  EXCLUDE_FROM_ALL
)
# Prevent overriding the parent project's compiler/linker settings on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_executable(ESIRuntimeCppTests
  ESIRuntimeTest.cpp
  ESIRuntimeValuesTest.cpp
  ESIRuntimeTypedPortsTest.cpp
)

# Place the binary where the pytest wrapper expects it — under
# <runtime-build-root>/tests/cpp/ — rather than the global bin/ directory.
set_target_properties(ESIRuntimeCppTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(ESIRuntimeCppTests PRIVATE
  ESICppRuntime
  GTest::gtest_main
)

if(NOT MSVC)
  target_link_options(ESIRuntimeCppTests PRIVATE -pthread)
endif()

add_dependencies(ESIRuntime ESIRuntimeCppTests)
