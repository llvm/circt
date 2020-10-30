find_package(CapnProto CONFIG PATHS "${CMAKE_SOURCE_DIR}/ext")
if(ESI_COSIM)
  add_library(EsiCosimDpiServer SHARED
    DpiEntryPoints.cpp
    Server.cpp
    EndPoint.cpp)

  add_dependencies(EsiCosimDpiServer CosimCapnp)
  target_link_libraries(EsiCosimDpiServer PRIVATE
      CapnProto::kj CapnProto::kj-async CapnProto::kj-gzip
      CapnProto::capnp CapnProto::capnp-rpc 
      MtiPli CosimCapnp)

  target_include_directories(EsiCosimDpiServer PRIVATE ${CAPNPC_OUTPUT_DIR})
  target_include_directories(EsiCosimDpiServer PRIVATE ${CAPNP_INCLUDE_DIRS})
  target_include_directories(EsiCosimDpiServer PRIVATE ${CIRCT_INCLUDE_DIR})
endif()
