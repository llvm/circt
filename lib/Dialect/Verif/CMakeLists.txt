add_circt_dialect_library(CIRCTVerif
  VerifDialect.cpp
  VerifOps.cpp
  VerifOpInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Verif

  DEPENDS
  MLIRVerifIncGen
  MLIRVerifOpInterfacesIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTLTL
  CIRCTSeq
  MLIRIR
  MLIRTransforms
)

add_subdirectory(Transforms)
