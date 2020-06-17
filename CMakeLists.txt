cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(circt LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")
if (MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

# CIRCT project.
set(CIRCT_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include ) # --src-root
set(CIRCT_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include ) # --includedir

set(CIRCT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CIRCT_BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
set(CIRCT_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

# Installing the headers and docs needs to depend on generating any public
# tablegen'd targets.
add_custom_target(circt-headers)
set_target_properties(circt-headers PROPERTIES FOLDER "Misc")
add_custom_target(circt-doc)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
#include_directories( ${MLIR_BINARY_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_subdirectory(include/circt)
add_subdirectory(lib)
add_subdirectory(tools)
#add_subdirectory(unittests)
add_subdirectory(test)

#option(CIRCT_INCLUDE_DOCS "Generate build targets for the CIRCT docs."
#  ${LLVM_INCLUDE_DOCS} ${MLIR_INCLUDE_DOCS})
#if (CIRCT_INCLUDE_DOCS)
#  add_subdirectory(docs)
#endif()

install(DIRECTORY include/circt
  DESTINATION include
  COMPONENT circt-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "LICENSE.TXT"
  )

install(DIRECTORY ${CIRCT_INCLUDE_DIR}/circt
  DESTINATION include
  COMPONENT circt-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "config.h" EXCLUDE
  )

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-circt-headers
                           DEPENDS circt-headers
                           COMPONENT circt-headers)
endif()

add_subdirectory(cmake/modules)
