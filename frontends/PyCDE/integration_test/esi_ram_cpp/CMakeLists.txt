cmake_minimum_required(VERSION 3.13.4)
project(esi_ram_test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Assert that CIRCT_DIR is defined.
if(NOT DEFINED CIRCT_DIR)
  message(FATAL_ERROR "CIRCT_DIR must be defined.")
endif()

if(DEFINED CAPNP_PATH)
  set(ENV{PKG_CONFIG_PATH}
    "${CAPNP_PATH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  find_package(CapnProto CONFIG PATHS ${CAPNP_PATH})
  else()
  set(ENV{PKG_CONFIG_PATH}
    "${CIRCT_DIR}/ext/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  find_package(CapnProto CONFIG PATHS "${CIRCT_DIR}/ext")
endif()

if (NOT CapnProto_FOUND)
  message(FATAL_ERROR "Cap'n Proto not found.")
endif()

capnp_generate_cpp(
    ESI_RAM_SRCS
    ESI_RAM_HDRS
    ${CMAKE_CURRENT_SOURCE_DIR}/hw/schema.capnp
)

add_executable(esi_ram_test
  esi_ram.cpp
  ${ESI_RAM_SRCS}
  ${ESI_RAM_HDRS}
)
target_link_libraries(esi_ram_test ${CAPNP_LIBRARIES})


message("ESI_RAM_SRCS: ${ESI_RAM_SRCS}")
message("ESI_RAM_HDRS: ${ESI_RAM_HDRS}")

target_include_directories(
    esi_ram_test
    PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CAPNP_INCLUDE_DIRS}
)
