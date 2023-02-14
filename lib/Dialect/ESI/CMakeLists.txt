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
  ESIServices.cpp
  ESIStdServices.cpp
  ESITranslations.cpp
  ESITypes.cpp
)

set(ESI_LinkLibs
  CIRCTSupport
  CIRCTComb
  CIRCTMSFT
  CIRCTMSFTTransforms
  CIRCTSV
  CIRCTHW
  MLIRIR
  MLIRTransforms
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTranslateLib
)

set(ESI_Deps
  ${ESI_LinkLibs}
  MLIRESITransformsIncGen
  MLIRESIInterfacesIncGen
)

set(CP_FILES
  ESIPrimitives.sv
)

if(CapnProto_FOUND)
  option(ESI_CAPNP "Enable ESI Capnp features" ON)
endif()

if (ESI_CAPNP)
  list(APPEND srcs capnp/Schema.cpp)
  list(APPEND ESI_LinkLibs CapnProto::capnp CapnProto::capnpc)
  list(APPEND CP_FILES ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/ESI/cosim/CosimDpi.capnp)
  list(APPEND CP_FILES cosim/Cosim_DpiPkg.sv)
  list(APPEND CP_FILES cosim/Cosim_Endpoint.sv)
  install(FILES ${CapnProto_capnp_IMPORTED_LOCATION} DESTINATION lib)
  install(FILES ${CapnProto_capnpc_IMPORTED_LOCATION} DESTINATION lib)
  install(FILES ${CapnProto_kj_IMPORTED_LOCATION} DESTINATION lib)
endif()

add_circt_dialect_library(CIRCTESI
  ${srcs}

  DEPENDS
  MLIRESITransformsIncGen
  MLIRESIEnumsIncGen
  ${ESI_Deps}

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC
  ${ESI_LinkLibs}
  )


foreach(CP_FILE ${CP_FILES})
  get_filename_component(DST ${CP_FILE} NAME)
  get_filename_component(SRC ${CP_FILE} REALPATH)
  add_custom_command(TARGET CIRCTESI PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E
    copy ${SRC} ${CIRCT_BINARY_DIR}/include/circt/Dialect/ESI/${DST})
endforeach()

if (ESI_CAPNP)
  target_compile_definitions(obj.CIRCTESI PRIVATE CAPNP)
  target_link_libraries(obj.CIRCTESI CapnProto::capnp CapnProto::capnpc)
endif()

add_subdirectory(cosim)
