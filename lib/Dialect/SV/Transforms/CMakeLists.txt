add_circt_dialect_library(CIRCTSVTransforms
  GeneratorCallout.cpp
  HWCleanup.cpp
  HWStubExternalModules.cpp
  HWLegalizeModules.cpp
  HWLegalizeNames.cpp
  HWMemSimImpl.cpp
  PrettifyVerilog.cpp
  SVExtractTestCode.cpp

  DEPENDS
  CIRCTSVTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
