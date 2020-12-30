##===- CMakeLists.txt - ESI code which needs capnp ------------*- cmake -*-===//
##
## Parts of the ESI dialect code need access to libcapnp and libcapnpc, which
## may not be available. Put all that code in here to make it an optional
## feature.
##
##===----------------------------------------------------------------------===//

if(CapnProto_FOUND)
  option(ESI_CAPNP "Enable ESI Capnp features" ON)

  add_circt_library(ESICapnp
    Schema.cpp

    ADDITIONAL_HEADER_DIRS PRIVATE
    ${CAPNP_INCLUDE_DIRS}
   
    LINK_LIBS PUBLIC
    CapnProto::capnp CapnProto::capnpc
    )
endif()
