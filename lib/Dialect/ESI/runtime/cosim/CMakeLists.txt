##===- CMakeLists.txt - ESI cosim support ---------------------*- cmake -*-===//
##
##===----------------------------------------------------------------------===//

# Compile Capnp file.
add_definitions(${CAPNP_DEFINITIONS})
set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/../cpp/include/backends)
include_directories(${CAPNPC_OUTPUT_DIR})
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})
capnp_generate_cpp(
  COSIM_CAPNP_SRCS COSIM_CANPN_HDRS
  "CosimDpi.capnp"
)

# Compile a library for ESI cosim capnp for both the API runtime backend and the
# cosim DPI server to use.
add_library(EsiCosimCapnp SHARED
  ${COSIM_CAPNP_HDRS}
  ${COSIM_CAPNP_SRCS}
  ${COSIM_SCHEMA_HDR}
)
target_include_directories(EsiCosimCapnp PUBLIC ${CAPNPC_OUTPUT_DIR})
target_include_directories(EsiCosimCapnp PUBLIC ${CAPNP_INCLUDE_DIRS})
target_link_libraries(EsiCosimCapnp PUBLIC
  CapnProto::kj CapnProto::kj-async CapnProto::kj-gzip
  CapnProto::capnp CapnProto::capnp-rpc 
)
install(TARGETS EsiCosimCapnp
  DESTINATION lib
  COMPONENT ESIRuntime
)

set(cosim_collateral
  Cosim_DpiPkg.sv
  Cosim_Endpoint.sv
  Cosim_Manifest.sv
  Cosim_MMIO.sv

  driver.sv
  driver.cpp
)

install(FILES
  ${cosim_collateral}
  DESTINATION cosim
  COMPONENT ESIRuntime
)

add_custom_target(esi-cosim
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/esi-cosim.py
          ${CIRCT_TOOLS_DIR}/esi-cosim.py)
foreach (cf ${cosim_collateral})
  add_custom_command(TARGET esi-cosim POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_CURRENT_SOURCE_DIR}/${cf}
              ${CIRCT_TOOLS_DIR}/../cosim/${cf}
  )
endforeach()
install(FILES
  esi-cosim.py
  DESTINATION bin
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
              GROUP_EXECUTE GROUP_READ
              WORLD_EXECUTE WORLD_READ
  COMPONENT ESIRuntime
)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
add_subdirectory(cosim_dpi_server)
add_subdirectory(MtiPliStub)
