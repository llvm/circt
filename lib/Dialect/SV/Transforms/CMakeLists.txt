add_circt_dialect_library(CIRCTSVTransforms
  GeneratorCallout.cpp
  PrettifyVerilog.cpp
  RTLCleanup.cpp
  RTLStubExternalModules.cpp
  RTLLegalizeNames.cpp
  RTLMemSimImpl.cpp

  DEPENDS
  CIRCTSVTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
