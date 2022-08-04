
add_circt_translation_library(CIRCTExportSystemC
  EmissionPrinter.cpp
  EmissionPattern.cpp
  ExportSystemC.cpp
  Patterns/CombEmissionPatterns.cpp
  Patterns/HWEmissionPatterns.cpp
  Patterns/SCFEmissionPatterns.cpp
  Patterns/SystemCEmissionPatterns.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSystemC
  CIRCTSupport
  MLIRIR
  MLIRTranslateLib
  )
