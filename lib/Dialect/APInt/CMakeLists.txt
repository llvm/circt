add_circt_dialect_library(CIRCTAPInt
  APIntOps.cpp
  APIntDialect.cpp
  APIntTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/APInt

  DEPENDS
  MLIRAPIntIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

  add_dependencies(circt-headers MLIRAPIntIncGen)
