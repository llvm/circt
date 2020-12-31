add_subdirectory(capnp)

set(ESI_Deps
  CIRCTSupport
  MLIRESIEnumsIncGen
)

set(ESI_LinkLibs
  CIRCTSupport
  CIRCTSV
  MLIREDSC
  MLIRIR
  MLIRTransforms
)

if (ESI_CAPNP)
  list(APPEND ESI_Deps ESICapnp)
  list(APPEND ESI_LinkLibs ESICapnp)
endif()

add_circt_dialect_library(CIRCTESI
  ESIDialect.cpp
  ESIOps.cpp
  ESIPasses.cpp
  ESITranslations.cpp
  ESITypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/ESI

  DEPENDS
  ${ESI_Deps}

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC
  ${ESI_LinkLibs}
  )

if (ESI_CAPNP)
  target_compile_definitions(obj.CIRCTESI PRIVATE CAPNP)
endif()

add_subdirectory(cosim)
