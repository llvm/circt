# Driven by `tests/unit/test_codegen.py`. Inputs:
#
#   CODEGEN_HARNESS_GENERATED_DIR  - directory holding the freshly
#                                    generated `codegen_harness/types.h`.
#   ESI_RUNTIME_LIB                - absolute path to the artifact to
#                                    link against. On Unix this is the
#                                    shared object (`.so` / `.dylib`);
#                                    on Windows it's the MSVC import
#                                    library (`.lib`) — linking a `.dll`
#                                    directly does not work with MSVC,
#                                    which is why we split the link vs
#                                    runtime artifacts.
#   ESI_RUNTIME_DLL                - Windows only, optional. Absolute
#                                    path to `ESICppRuntime.dll` to
#                                    stage next to the executable so it
#                                    runs without modifying `PATH`.
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
    "Absolute path to the ESI runtime link artifact (.so/.dylib/.lib)")
set(ESI_RUNTIME_DLL "" CACHE FILEPATH
    "Absolute path to ESICppRuntime.dll (Windows runtime staging only)")
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

if(UNIX)
  get_filename_component(_esi_runtime_lib_dir "${ESI_RUNTIME_LIB}" DIRECTORY)
  set_target_properties(codegen_harness PROPERTIES
    BUILD_RPATH "${_esi_runtime_lib_dir}")
elseif(WIN32)
  # Prefer the explicit ESI_RUNTIME_DLL when one was provided (MSVC link
  # via .lib, runtime via .dll). Fall back to ESI_RUNTIME_LIB itself for
  # the MinGW case where linking is done directly against the .dll.
  if(ESI_RUNTIME_DLL)
    set(_esi_runtime_dll "${ESI_RUNTIME_DLL}")
  else()
    set(_esi_runtime_dll "${ESI_RUNTIME_LIB}")
  endif()
  add_custom_command(TARGET codegen_harness POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${_esi_runtime_dll}"
      "$<TARGET_FILE_DIR:codegen_harness>")
endif()
