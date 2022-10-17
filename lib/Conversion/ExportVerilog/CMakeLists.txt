
add_circt_translation_library(CIRCTExportVerilog
  ApplyLoweringOptions.cpp
  ExportVerilog.cpp
  LegalizeNames.cpp
  PrepareForEmission.cpp
  RearrangableOStream.cpp
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
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransforms
  )
