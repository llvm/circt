add_circt_dialect_library(CIRCTESI
  ESIDialect.cpp
  ESIOps.cpp
  ESIPasses.cpp
  ESITypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/ESI

  DEPENDS
  CIRCTSupport
  MLIRESIIncGen
  MLIRESIEnumsIncGen

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC
  CIRCTSupport
  CIRCTSV
  MLIREDSC
  MLIRIR
  MLIRTransforms
  )


add_dependencies(circt-headers MLIRESIIncGen)

add_subdirectory(cosim)
