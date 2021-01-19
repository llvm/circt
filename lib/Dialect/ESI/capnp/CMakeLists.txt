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

    DEPENDS
    CIRCTSupport
    MLIRESIEnumsIncGen

    LINK_COMPONENTS
    Core
    Support

    LINK_LIBS PUBLIC
    CIRCTSupport
    CIRCTSV
    CIRCTRTL
    MLIRIR
    CapnProto::capnp CapnProto::capnpc
  )
  target_link_libraries(obj.ESICapnp CapnProto::capnp CapnProto::capnpc)
endif()
