add_circt_dialect_library(CIRCTESI
  ESIDialect.cpp
  ESIOps.cpp
  ESIPasses.cpp
  ESITypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/ESI

  DEPENDS
  MLIRESIIncGen
  MLIRESIEnumsIncGen
  CIRCTSupport

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  CIRCTSV
  MLIRTransforms
  CIRCTSupport
  )


add_dependencies(circt-headers MLIRESIIncGen)

add_subdirectory(cosim)
