
add_circt_translation_library(CIRCTExportVerilog
  ApplyLoweringOptions.cpp
  ExportVerilog.cpp
  LegalizeAnonEnums.cpp
  LegalizeNames.cpp
  PrepareForEmission.cpp
  PruneZeroValuedLogic.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransforms
)
