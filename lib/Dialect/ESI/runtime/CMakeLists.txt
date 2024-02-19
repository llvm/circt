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
project(ESIRuntime)
include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# We need zlib to uncompress the manifest.
find_package(ZLIB REQUIRED)

# JSON parser for the manifest.
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.2
)
FetchContent_MakeAvailable(json)

include_directories(cpp/include)

set(ESIRuntimeSources
  cpp/lib/Accelerator.cpp
  cpp/lib/Context.cpp
  cpp/lib/Common.cpp
  cpp/lib/Design.cpp
  cpp/lib/Manifest.cpp
  cpp/lib/Services.cpp
  cpp/lib/Ports.cpp
  cpp/lib/Utils.cpp

  cpp/lib/backends/Trace.cpp
)
set(ESIRuntimeLinkLibraries
  ZLIB::ZLIB
  nlohmann_json::nlohmann_json
)
set(ESIPythonRuntimeSources
  python/esi/__init__.py
  python/esi/accelerator.py
  python/esi/types.py
  python/esi/esiCppAccel.pyi
)
set(ESIRuntimeIncludeDirs)
set(ESIRuntimeCxxFlags)
set(ESIRuntimeLinkFlags)
set(ESIRuntimeLibDirs)

IF(MSVC)
    set(CMAKE_CXX_FLAGS "/EHa")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
ENDIF(MSVC)

# If Cap'nProto hasn't been explicitly disabled, find it.
option(CAPNP_DISABLE "Disable Cap'nProto (needed for cosimulation).")
if (CAPNP_DISABLE)
  message (STATUS "Disabling Cap'nProto (needed for cosimulation).")
elseif(NOT CapnProto_FOUND) # Check if someone else already imported.
  if(DEFINED CAPNP_PATH)
    set(ENV{PKG_CONFIG_PATH}
      "${CAPNP_PATH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    find_package(CapnProto CONFIG PATHS ${CAPNP_PATH})
  else()
    set(ENV{PKG_CONFIG_PATH}
      "${CMAKE_CURRENT_SOURCE_DIR}/ext/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    find_package(CapnProto CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/ext")
  endif()
endif()

# If Cap'nProto has been found, generate the headers and definitions.
if(CapnProto_FOUND)
  set(ESI_COSIM true)
  message("-- ESI cosim enabled")

  message(STATUS "Found Cap'nProto at ${CapnProto_DIR}.")
  set(CMAKE_INSTALL_RPATH ${capnp_LIBDIR})
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

  add_subdirectory(cosim)

  set(ESIRuntimeSources
    ${ESIRuntimeSources}
    cpp/lib/backends/Cosim.cpp
  )
  set(ESIRuntimeLinkLibraries
    ${ESIRuntimeLinkLibraries}
    EsiCosimCapnp
  )
endif()

option(XRT_PATH "Path to XRT lib.")
if (XRT_PATH)
  message("-- XRT enabled with path ${XRT_PATH}")

  set(ESIRuntimeSources
    ${ESIRuntimeSources}
    cpp/lib/backends/Xrt.cpp
  )
  set(ESIRuntimeIncludeDirs
    ${ESIRuntimeIncludeDirs}
    ${XRT_PATH}/include
  )
  set(ESIRuntimeCxxFlags
    ${ESIRuntimeCxxFlags}
    -fmessage-length=0
    -Wno-nested-anon-types
    -Wno-c++98-compat-extra-semi
  )
  set(ESIRuntimeLinkLibraries
    ${ESIRuntimeLinkLibraries}
    xrt_coreutil
  )
  set(ESIRuntimeLinkFlags
    ${ESIRuntimeLinkFlags}
    -pthread
  )
  set(ESIRuntimeLibDirs
    ${ESIRuntimeLibDirs}
    ${XRT_PATH}/lib
  )
endif()

# The core API. For now, compile the backends into it directly.
# TODO: make this a plugin architecture.
add_library(ESIRuntime SHARED
  ${ESIRuntimeSources}
)
target_link_libraries(ESIRuntime PRIVATE ${ESIRuntimeLinkLibraries})
target_include_directories(ESIRuntime PRIVATE ${ESIRuntimeIncludeDirs})
target_compile_options(ESIRuntime PRIVATE ${ESIRuntimeCxxFlags})
target_link_directories(ESIRuntime PRIVATE ${ESIRuntimeLibDirs})
target_link_options(ESIRuntime PRIVATE ${ESIRuntimeLinkFlags})

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(ESIRuntime PRIVATE -Wno-covered-switch-default)
endif()

# The esiquery tool is a simple wrapper around the SysInfo API.
add_executable(esiquery
  cpp/tools/esiquery.cpp
)
target_link_libraries(esiquery PRIVATE ESIRuntime)

# Global variable for the path to the ESI runtime for use by tests.
set(ESIRuntimePath "${CMAKE_CURRENT_BINARY_DIR}"
  CACHE INTERNAL "Path to ESI runtime" FORCE)

# Pybind11 is used to wrap the ESIRuntime APIs.
find_package(Python COMPONENTS Interpreter Development)
if(Python_FOUND)
  IF(MSVC)
    # Work around an issue with pybind11 and cmake incompatibility on Windows in debug mode.
    set_target_properties(Python::Module PROPERTIES
          MAP_IMPORTED_CONFIG_DEBUG ";RELEASE")
  ENDIF(MSVC)

  if(pybind11_DIR)
    message(STATUS "Using explicit pybind11 cmake directory: ${pybind11_DIR} (-Dpybind11_DIR to change)")
  else()
    message(STATUS "Checking for pybind11 in python path...")
    execute_process(
      COMMAND "${Python_EXECUTABLE}"
      -c "import pybind11;print(pybind11.get_cmake_dir(), end='')"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE STATUS
      OUTPUT_VARIABLE PACKAGE_DIR
      ERROR_QUIET)
    if(NOT STATUS EQUAL "0")
      message(FATAL_ERROR "pybind11 not found (install via 'pip install pybind11' or set pybind11_DIR)")
    endif()
    message(STATUS "found (${PACKAGE_DIR})")
    set(pybind11_DIR "${PACKAGE_DIR}")
  endif()

  # Now, find pybind11.
  find_package(pybind11 CONFIG)
  if (NOT pybind11_FOUND)
    message (STATUS "Could not find pybind11. Disabling Python API.")
  else()
    # Compile Pybind11 module and copy to the correct python directory.
    pybind11_add_module(esiCppAccel python/esi/esiCppAccel.cpp)
    target_link_libraries(esiCppAccel PRIVATE ESIRuntime)
    add_custom_command(
        TARGET esiCppAccel
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:esiCppAccel>
            "${CMAKE_CURRENT_BINARY_DIR}/python/esi/$<TARGET_FILE_NAME:esiCppAccel>"
    )

    # Copy each of the Python sources to the build dir.
    foreach(pysrc ${ESIPythonRuntimeSources})
      add_custom_command(
          OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${pysrc}
          COMMAND ${CMAKE_COMMAND} -E copy
              ${CMAKE_CURRENT_SOURCE_DIR}/${pysrc}
              ${CMAKE_CURRENT_BINARY_DIR}/${pysrc}
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${pysrc}
      )
    endforeach()

    # Custom target for the Python runtime just aggregates the python sources
    # and Pybind11 module.
    add_custom_target(ESIPythonRuntime
      DEPENDS
        ${ESIPythonRuntimeSources}
        esiCppAccel
    )

  endif()
endif()
