##===- CMakeLists.txt - ESI cosim support ---------------------*- cmake -*-===//
##
##===----------------------------------------------------------------------===//

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

add_library(EsiCosimGRPC SHARED "${CMAKE_CURRENT_LIST_DIR}/cosim.proto")
install(TARGETS EsiCosimGRPC DESTINATION lib COMPONENT ESIRuntime)
target_link_libraries(EsiCosimGRPC PUBLIC protobuf::libprotobuf gRPC::grpc++)
set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_include_directories(EsiCosimGRPC PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")

protobuf_generate(
    TARGET EsiCosimGRPC
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")
protobuf_generate(
    TARGET EsiCosimGRPC
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=\$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(CosimRpcServer SHARED
  lib/RpcServer.cpp
)
target_link_libraries(CosimRpcServer
  PUBLIC
  EsiCosimGRPC
  ESIRuntime
)
install(TARGETS CosimRpcServer DESTINATION lib COMPONENT ESIRuntime)

add_subdirectory(cosim_dpi_server)
add_subdirectory(MtiPliStub)
