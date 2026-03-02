add_circt_dialect_library(CIRCTOpLib
  OpLibDialect.cpp
  OpLibOps.cpp
  OpLibAttributes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/OpLib

  LINK_LIBS PUBLIC
  MLIRIR
  CIRCTSupport

  DEPENDS
  CIRCTSupport
  MLIROpLibIncGen
  CIRCTOpLibAttributesIncGen
  CIRCTCalyx
)
