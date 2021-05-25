add_circt_dialect_library(CIRCTSVTransforms
  GeneratorCallout.cpp
  HWCleanup.cpp
  HWStubExternalModules.cpp
  HWLegalizeNames.cpp
  HWMemSimImpl.cpp
  HWDeclareTypedecls.cpp
  PrettifyVerilog.cpp

  DEPENDS
  CIRCTSVTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
