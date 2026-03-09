add_circt_translation_library(CIRCTImportLiberty
  ImportLiberty.cpp

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
