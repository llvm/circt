##===- CMakeLists.txt - ESI runtime CMake ---------------------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
## Compile definitions for the ESI runtime. Distributed with an ESI compiler as
## part of the ESI collateral. For now, we require that users compile this
## themselves since it needs to be compiled for each Python version and OS then
## packed together. Eventually, we'll just be distributing (lots of) binaries.
##
## We require Python development package and pybind11 to compile the Python API.
##
## ESI cosimulation requires Cap'nProto as we use it for our RPC with the
## simulator. It must be fetched separately, but is optional if you don't want
## cosimulation.
##
## DO NOT EDIT!
## This file is distributed as part of an ESI package. The source for this file
## should always be modified within CIRCT.
##
##===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.20)
project(ESIRuntime LANGUAGES CXX)
include(FetchContent)

set(ESI_STATIC_RUNTIME OFF CACHE BOOL "Build the ESI runtime as a static library.")
if(ESI_STATIC_RUNTIME)
  message("-- Building ESI runtime as a static library.")
endif()

option(ESI_RUNTIME_TRACE "Enable ESI trace-level logging." OFF)
if (ESI_RUNTIME_TRACE)
  message("-- ESI runtime trace-level logging enabled.")
  add_compile_definitions(ESI_RUNTIME_TRACE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_BUILD_RPATH
      "${CMAKE_BUILD_RPATH}:${CMAKE_BINARY_DIR}/lib:$ORIGIN/../lib")

# JSON parser for the manifest.
if (NOT TARGET nlohmann_json)
  message("-- ESI runtime pulling down json")
  FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
  )
  FetchContent_MakeAvailable(json)
endif()

# We need zlib to uncompress the manifest. We build it statically to avoid
# conflicts with local zlib versions, and since we don't export anything zlib
# related in our public headers.
message("-- pulling down zlib from git")
set(ZLIB_BUILD_SHARED OFF)
set(ZLIB_BUILD_STATIC ON)
set(ZLIB_BUILD_TESTING OFF)
FetchContent_Declare(
  ZLIB
  GIT_REPOSITORY https://github.com/madler/zlib.git
  # Note @mortbopet: The latest zlib release (v1.3.1) is fairly old (Jan 22. 2024)
  # and is missing some key changes/fixes for building w/ a static library.
  # For now, fixed by tagging to head of their development branch.
  # Change once a new release is made that includes these fixes.
  GIT_TAG        5a82f71ed1dfc0bec044d9702463dbdf84ea3b71
)
FetchContent_MakeAvailable(ZLIB)
set(ZLIB_INCLUDE_DIR ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
if(UNIX)
  target_compile_options(zlibstatic
    PRIVATE
      -fPIC
  )
endif()

# fmt is used for logging and formatting messages.
if (NOT TARGET fmt::fmt-header-only)
  message(STATUS "fmt not found, pulling down from git")
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG        11.1.4)
  FetchContent_MakeAvailable(fmt)
endif()

# CLI11 is used for tools' command line options parsing.
if (NOT TARGET CLI11::CLI11)
  message(STATUS "CLI11 not found, pulling down from git")
  FetchContent_Declare(
      cli11_proj
      QUIET
      GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
      GIT_TAG v2.5.0
  )
  FetchContent_MakeAvailable(cli11_proj)
endif()

# In a Python wheel build, we need to install libraries to different places.
option(WHEEL_BUILD "Set up the build for a Python wheel." OFF)
if (WHEEL_BUILD)
  message(STATUS "Setting up for a Python wheel build.")
endif()

##===----------------------------------------------------------------------===//
## Overall target to build everything.
##===----------------------------------------------------------------------===//
add_custom_target(ESIRuntime)

##===----------------------------------------------------------------------===//
## Core ESI runtime.
##===----------------------------------------------------------------------===//

set(ESICppRuntimeSources
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/Accelerator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/Context.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/Common.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/Design.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/Engines.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/Logging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/Manifest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/Services.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/Ports.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/Types.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/Utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/Values.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/backends/Trace.cpp
)
set(ESICppRuntimeHeaders
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/Utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/Accelerator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/CLI.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/Common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/Context.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/Design.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/Engines.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/Logging.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/Manifest.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/Values.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/Types.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/Ports.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/Services.h
)
set(ESICppRuntimeBackendHeaders
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/backends/Trace.h
)

IF(MSVC)
    # ESI runtime requires exceptions. Purge any exception-related flags from
    # CXX_FLAGS, and add /EHa.
    string(REGEX REPLACE "/EH[a-zA-Z-]*" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHa")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
ENDIF(MSVC)

if(ESI_STATIC_RUNTIME)
  add_library(ESICppRuntime STATIC
    ${ESICppRuntimeSources}
  )
else()
  add_library(ESICppRuntime SHARED
    ${ESICppRuntimeSources}
  )
endif()
add_library(esiaccel::ESICppRuntime ALIAS ESICppRuntime)

target_include_directories(ESICppRuntime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)
if (DEFINED ZLIB_INCLUDE_DIR)
  target_include_directories(ESICppRuntime PRIVATE
    ${ZLIB_INCLUDE_DIR})
endif()

target_link_libraries(ESICppRuntime PRIVATE
  ZLIB::ZLIBSTATIC
  nlohmann_json::nlohmann_json
)
target_link_libraries(ESICppRuntime PRIVATE
  fmt::fmt-header-only
)
if(NOT MSVC)
  target_link_libraries(ESICppRuntime PRIVATE
    dl
  )
  target_link_options(ESICppRuntime PUBLIC
    -pthread
  )
endif()
add_dependencies(ESIRuntime ESICppRuntime)
if (WIN32)
  set(ESIRT_INSTALL_BINDIR ".")
  set(ESIRT_INSTALL_LIBDIR ".")
else()
  set(ESIRT_INSTALL_BINDIR "bin")
  set(ESIRT_INSTALL_LIBDIR "lib")
endif()

install(TARGETS ESICppRuntime
  DESTINATION ${ESIRT_INSTALL_LIBDIR}
  RUNTIME_DEPENDENCIES
    PRE_EXCLUDE_REGEXES .*
  COMPONENT ESIRuntime
)

install(FILES ${ESICppRuntimeHeaders}
  DESTINATION include/esi
  COMPONENT ESIRuntime
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cmake/esiaccelConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cpp/cmake/esiaccelConfig.cmake
  @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cpp/cmake/esiaccelConfig.cmake
  DESTINATION cmake
  COMPONENT ESIRuntime
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(ESICppRuntime PRIVATE -Wno-covered-switch-default)
endif()

# Global variable for the path to the ESI runtime for use by tests.
set(ESIRuntimePath "${CMAKE_CURRENT_BINARY_DIR}"
  CACHE INTERNAL "Path to ESI runtime" FORCE)


##===----------------------------------------------------------------------===//
## The esiquery tool is a simple wrapper around the SysInfo API.
##===----------------------------------------------------------------------===//

add_executable(esiquery
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tools/esiquery.cpp
)
target_link_libraries(esiquery PRIVATE
  ESICppRuntime
  CLI11::CLI11
)
add_dependencies(ESIRuntime esiquery)
install(TARGETS esiquery
  DESTINATION ${ESIRT_INSTALL_BINDIR}
  COMPONENT ESIRuntime
)

##===----------------------------------------------------------------------===//
## The esitester tool is both an example and test driver. As it is not intended
## for production use, it is not installed.
##===----------------------------------------------------------------------===//

add_executable(esitester
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tools/esitester.cpp
)
target_link_libraries(esitester PRIVATE
  ESICppRuntime
  CLI11::CLI11
)
add_dependencies(ESIRuntime esitester)

# Add the Python bindings.
add_subdirectory(python)

##===----------------------------------------------------------------------===//
## Backends are loaded dynamically as plugins.
##===----------------------------------------------------------------------===//

option(ESI_COSIM "Enable ESI cosimulation." ON)
if(ESI_COSIM)
  message("-- ESI cosim enabled")

  # gRPC for cosimulation. Local install required.
  option(GRPC_PATH "Location of gRPC install.")
  if (${GRPC_PATH})
    find_package(Protobuf REQUIRED CONFIG HINTS ${GRPC_PATH})
    find_package(gRPC REQUIRED CONFIG HINTS ${GRPC_PATH})
  else()
    find_package(Protobuf REQUIRED CONFIG)
    find_package(gRPC REQUIRED CONFIG)
  endif()

  add_library(CosimBackend SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/backends/Cosim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/backends/RpcServer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cosim.proto
  )
  set(ESICppRuntimeBackendHeaders
    ${ESICppRuntimeBackendHeaders}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/backends/Cosim.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/backends/RpcServer.h
  )

  target_link_libraries(CosimBackend PUBLIC
    ESICppRuntime
    protobuf::libprotobuf
    gRPC::grpc++
  )
  set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
  target_include_directories(CosimBackend PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")
  protobuf_generate(
      TARGET CosimBackend
      PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")
  protobuf_generate(
      TARGET CosimBackend
      LANGUAGE grpc
      GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
      PLUGIN "protoc-gen-grpc=\$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
      PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

  add_dependencies(ESIRuntime CosimBackend)

  if(WIN32)
    # Windows cosim backend carries a lot of runtime dependencies, mainly
    # transitively through gRPC. Make sure they're installed as well.
    install(TARGETS CosimBackend
      DESTINATION ${ESIRT_INSTALL_LIBDIR}
      RUNTIME_DEPENDENCIES
        PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
        POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
      COMPONENT ESIRuntime
    )
  else()
    install(TARGETS CosimBackend
      DESTINATION ${ESIRT_INSTALL_LIBDIR}
      COMPONENT ESIRuntime
    )
  endif()

  # Build the RTL DPI cosim server.
  add_subdirectory(cosim_dpi_server)
else()
  message("-- ESI cosim disabled")
endif()

option(XRT_PATH "Path to XRT lib.")
if (XRT_PATH)
  message("-- XRT enabled with path ${XRT_PATH}")

  add_library(XrtBackend SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/lib/backends/Xrt.cpp
  )
  set(ESICppRuntimeBackendHeaders
    ${ESICppRuntimeBackendHeaders}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/esi/backends/Xrt.h
  )
  target_include_directories(XrtBackend PRIVATE
    ${XRT_PATH}/include
  )
  target_compile_options(XrtBackend PRIVATE
    -fmessage-length=0
    -Wno-nested-anon-types
    -Wno-c++98-compat-extra-semi
  )
  target_link_libraries(XrtBackend PRIVATE
    ESICppRuntime
    xrt_coreutil
  )
  target_link_options(XrtBackend PRIVATE
    -pthread
    -L${XRT_PATH}/lib
  )
  add_dependencies(ESIRuntime XrtBackend)
  install(TARGETS XrtBackend
    DESTINATION ${ESIRT_INSTALL_LIBDIR}
    COMPONENT ESIRuntime
  )
endif()

install(FILES ${ESICppRuntimeBackendHeaders}
  DESTINATION include/esi/backends
  COMPONENT ESIRuntime
)
