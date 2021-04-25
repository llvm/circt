add_circt_dialect_library(CIRCTSVTransforms
  RTLCleanup.cpp
  RTLStubExternalModules.cpp
  RTLLegalizeNames.cpp
  GeneratorCallout.cpp
  RTLMemSimImpl.cpp

  DEPENDS
  CIRCTSVTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
