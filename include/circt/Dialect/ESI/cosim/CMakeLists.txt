##===- CMakeLists.txt - Cosim DPI library headers -------------*- cmake -*-===//
##
## Generate the Capnp headers if Capnp is found.
##
##===----------------------------------------------------------------------===//

if(CapnProto_FOUND)
  option(ESI_COSIM "Enable ESI Cosimulation" ON)
  message("-- Enabling ESI cosim")

  file(READ CosimDpi.capnp EsiCosimSchema)
  set(COSIM_SCHEMA_HDR ${CIRCT_BINARY_DIR}/include/circt/Dialect/ESI/CosimSchema.h)
  configure_file(CosimSchema.h.in ${COSIM_SCHEMA_HDR})

  if (MSVC)
    string(REPLACE "/EHs-c-" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHa")
  else ()
    string(REPLACE "-fno-exceptions" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  endif ()

  add_definitions(${CAPNP_DEFINITIONS})
  capnp_generate_cpp(COSIM_CAPNP_SRCS COSIM_CANPN_HDRS CosimDpi.capnp)
  add_library(EsiCosimCapnp
    ${COSIM_CAPNP_HDRS}
    ${COSIM_CAPNP_SRCS}
    ${COSIM_SCHEMA_HDR})
  target_link_libraries(EsiCosimCapnp
    PRIVATE
    CapnProto::capnp
    CapnProto::capnp-rpc)

  add_custom_command(TARGET EsiCosimCapnp PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E
    copy "${CMAKE_CURRENT_SOURCE_DIR}/CosimDpi.capnp" "CosimDpi.capnp")
  target_sources(EsiCosimCapnp PUBLIC "CosimDpi.capnp")
  install(FILES "CosimDpi.capnp" DESTINATION runtime)
endif()
