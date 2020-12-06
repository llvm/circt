##===- CMakeLists.txt - CIRCT cmake root ----------------------*- cmake -*-===//
##
## Configure the CIRCT build.
##
##===----------------------------------------------------------------------===//

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

# If we are not building as a part of LLVM, build Circt as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(circt LANGUAGES CXX C)
  
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED YES)
  
  option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
  option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)
  
  find_package(MLIR REQUIRED CONFIG)
  
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
  
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  set(CIRCT_BUILT_STANDALONE 1)
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")
else()
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include ) # --src-root
  set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include ) # --includedir
  set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
endif()

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

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
#include_directories( ${MLIR_BINARY_INCLUDE_DIR})
include_directories(${CIRCT_MAIN_SRC_DIR})
include_directories(${CIRCT_INCLUDE_DIR})

# If Verilator hasn't been explicitly disabled, find it.
option(VERILATOR_DISABLE "Disable the Verilator tests.")
if (VERILATOR_DISABLE)
  message(STATUS "Disabling Verilator tests.")
else()
  # Detect if Verilator is present.
  if (NOT DEFINED VERILATOR_PATH)
    find_program(VERILATOR_PATH "verilator" PATHS
      "${CMAKE_CURRENT_SOURCE_DIR}/ext/bin" NO_DEFAULT_PATH)
    find_program(VERILATOR_PATH "verilator")
  endif()

  if(EXISTS ${VERILATOR_PATH})
    message(STATUS "Found Verilator at ${VERILATOR_PATH}.")

    # Find Verilator version.
    execute_process(COMMAND ${VERILATOR_PATH} --version
      OUTPUT_VARIABLE VERILATOR_VERSION)
    string(REGEX MATCH "Verilator (([0-9]+)\.([0-9]+)) \.*"
      MATCH ${VERILATOR_VERSION})
    # It's gotta be at least v4.034.
    if (${CMAKE_MATCH_1} LESS 4.034)
      message(FATAL_ERROR "CIRCT only supports Verilator version 4.034 and up. \
                           Found version: ${CMAKE_MATCH_1}. You can disable \
                           the Verilator tests with '-DVERILATOR_DISABLE=ON'.")
      set(VERILATOR_PATH "")
    endif()
  else()
    set(VERILATOR_PATH "")
    message(STATUS "Did not find Verilator.")
  endif()
endif()

# If Yosys hasn't been explicitly disabled, find it.
option(YOSYS_DISABLE "Disable the yosys tests.")
if (YOSYS_DISABLE)
  message(STATUS "Disabling yosys tests.")
else()
  find_program(YOSYS_PATH "yosys")
  if(EXISTS ${YOSYS_PATH})
    message(STATUS "Found yosys at ${YOSYS_PATH}.")
  else()
    set(YOSYS_PATH "")
    message(STATUS "Did not find yosys.")
  endif()
endif()

# If capnp hasn't been explicitly disabled, find it.
option(CAPNP_DISABLE "Disable Cap'nProto (needed for cosimulation).")
if (CAPNP_DISABLE)
  message (STATUS "Disabling Cap'nProto.")
else()
  if(DEFINED CAPNP_PATH)
    set(ENV{PKG_CONFIG_PATH}
      "${CAPNP_PATH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    find_package(CapnProto CONFIG PATHS ${CAPNP_PATH})
  else()
    set(ENV{PKG_CONFIG_PATH}
      "${CMAKE_CURRENT_SOURCE_DIR}/ext/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    find_package(CapnProto CONFIG PATHS "${CMAKE_SOURCE_DIR}/ext")
  endif()
endif()

add_subdirectory(include/circt)
add_subdirectory(lib)
add_subdirectory(tools)
#add_subdirectory(unittests)
add_subdirectory(test)
add_subdirectory(integration_test)

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
  PATTERN "*.sv"
  PATTERN "LICENSE.TXT"
  )

install(DIRECTORY include/circt-c
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
