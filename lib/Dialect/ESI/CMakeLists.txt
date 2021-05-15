##===- CMakeLists.txt - ESI dialect code ----------------------*- cmake -*-===//
##
## Implementation files the ESI dialect. Optionally include the Capnp
## dependent code.
##
##===----------------------------------------------------------------------===//

set(srcs
  ESIDialect.cpp
  ESIFolds.cpp
  ESIOps.cpp
  ESIPasses.cpp
  ESITranslations.cpp
  ESITypes.cpp
)

set(ESI_LinkLibs
  CIRCTSupport
  CIRCTComb
  CIRCTSV
  CIRCTHW
  MLIREDSC
  MLIRIR
  MLIRTransforms
  MLIRTranslation
)

set(ESI_Deps
  ${ESI_LinkLibs}
  MLIRESIEnumsIncGen
)

if(CapnProto_FOUND)
  option(ESI_CAPNP "Enable ESI Capnp features" ON)
endif()

if (ESI_CAPNP)
  list(APPEND srcs capnp/Schema.cpp)
  list(APPEND ESI_LinkLibs CapnProto::capnp CapnProto::capnpc)
  install(FILES ${CapnProto_capnp_IMPORTED_LOCATION} DESTINATION lib)
  install(FILES ${CapnProto_capnpc_IMPORTED_LOCATION} DESTINATION lib)
  install(FILES ${CapnProto_kj_IMPORTED_LOCATION} DESTINATION lib)
endif()

add_circt_dialect_library(CIRCTESI
  ${srcs}

  DEPENDS
  MLIRESIEnumsIncGen
  ${ESI_Deps}

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC
  ${ESI_LinkLibs}
  )

if (ESI_CAPNP)
  target_compile_definitions(obj.CIRCTESI PRIVATE CAPNP)
  target_link_libraries(obj.CIRCTESI CapnProto::capnp CapnProto::capnpc)
endif()

add_subdirectory(cosim)
