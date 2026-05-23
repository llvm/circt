# Driven by `tests/unit/test_codegen.py`. Two inputs:
#
#   CODEGEN_HARNESS_GENERATED_DIR  - directory holding the freshly
#                                    generated `codegen_harness/types.h`.
#   ESI_RUNTIME_LIB                - absolute path to
#                                    `libESICppRuntime.{so,dylib,dll}` to
#                                    link against (resolved Python-side
#                                    from the in-tree `esiaccel` package).
#
# The in-tree `esi/` headers sit at a fixed relative path from this file.

cmake_minimum_required(VERSION 3.20)
project(codegen_harness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CODEGEN_HARNESS_GENERATED_DIR "" CACHE PATH
    "Root directory containing the generated `codegen_harness/types.h`")
set(ESI_RUNTIME_LIB "" CACHE FILEPATH
    "Absolute path to libESICppRuntime.{so,dylib,dll}")
if(NOT CODEGEN_HARNESS_GENERATED_DIR)
  message(FATAL_ERROR "CODEGEN_HARNESS_GENERATED_DIR must be set")
endif()
if(NOT ESI_RUNTIME_LIB)
  message(FATAL_ERROR "ESI_RUNTIME_LIB must be set")
endif()

add_executable(codegen_harness codegen_harness.cpp)
target_include_directories(codegen_harness PRIVATE
  "${CODEGEN_HARNESS_GENERATED_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include")
target_link_libraries(codegen_harness PRIVATE "${ESI_RUNTIME_LIB}")

get_filename_component(_esi_runtime_lib_dir "${ESI_RUNTIME_LIB}" DIRECTORY)
if(UNIX)
  set_target_properties(codegen_harness PROPERTIES
    BUILD_RPATH "${_esi_runtime_lib_dir}")
elseif(WIN32)
  # Stage the DLL next to the .exe so the harness runs without PATH edits.
  add_custom_command(TARGET codegen_harness POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ESI_RUNTIME_LIB}"
      "$<TARGET_FILE_DIR:codegen_harness>")
endif()
