add_circt_dialect_library(CIRCTCalyxLib
  CalyxLibDialect.cpp
  CalyxLibOps.cpp
  CalyxLibAttributes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/CalyxLib

  LINK_LIBS PUBLIC
  MLIRIR
  CIRCTSupport

  DEPENDS
  CIRCTSupport
  MLIRCalyxLibIncGen
  CIRCTCalyxLibAttributesIncGen
  CIRCTCalyx
)
