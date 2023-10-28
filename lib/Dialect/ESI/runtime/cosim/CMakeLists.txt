##===- CMakeLists.txt - ESI cosim support ---------------------*- cmake -*-===//
##
##===----------------------------------------------------------------------===//

# Cap'nProto MUST be built with exceptions enabled.
if (MSVC)
  string(REPLACE "/EHs-c-" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHa")
else ()
  string(REPLACE "-fno-exceptions" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif ()

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
add_library(EsiCosimCapnp STATIC
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

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
add_subdirectory(cosim_dpi_server)
add_subdirectory(MtiPliStub)
