##===- CMakeLists.txt - Cosim DPI library headers -------------*- cmake -*-===//
##
## Generate the Capnp headers if Capnp is found.
##
##===----------------------------------------------------------------------===//

if(CapnProto_FOUND)
  option(ESI_COSIM "Enable ESI Cosimulation" ON)
  message("-- Enabling ESI cosim")

  file(READ CosimDpi.capnp EsiCosimSchema)
  configure_file(CosimSchema.h.in
    ${CIRCT_BINARY_DIR}/include/circt/Dialect/ESI/CosimSchema.h)

  add_definitions(${CAPNP_DEFINITIONS})
  capnp_generate_cpp(COSIM_CAPNP_SRCS COSIM_CANPN_HDRS CosimDpi.capnp)
  add_library(EsiCosimCapnp
    ${COSIM_CAPNP_HDRS}
    ${COSIM_CAPNP_SRCS}
    CosimSchema.h)
  target_link_libraries(EsiCosimCapnp PRIVATE CapnProto::capnp)
endif()
