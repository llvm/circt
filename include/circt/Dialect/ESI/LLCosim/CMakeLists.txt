##===- CMakeLists.txt - ---------------------------------------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

if(CapnProto_FOUND)
  option(ESI_COSIM "Enable ESI Low Level Cosimulation" ON)
  message("-- Enabling ESI LLCosim")

  file(READ LLCosimDpi.capnp EsiLLCosimSchema)

  if (MSVC)
    string(REPLACE "/EHs-c-" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHa")
  else ()
    string(REPLACE "-fno-exceptions" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  endif ()

  add_definitions(${CAPNP_DEFINITIONS})
  capnp_generate_cpp(COSIM_CAPNP_SRCS COSIM_CANPN_HDRS LLCosimDpi.capnp)
  add_library(EsiLLCosimCapnp
    ${COSIM_CAPNP_HDRS}
    ${COSIM_CAPNP_SRCS}
    ${COSIM_SCHEMA_HDR})
  target_link_libraries(EsiLLCosimCapnp
    PRIVATE
    CapnProto::capnp
    CapnProto::capnp-rpc)
  target_sources(EsiLLCosimCapnp PUBLIC FILE_SET HEADERS
    BASE_DIRS "."
    FILES "LLCosimDpi.capnp")
endif()
