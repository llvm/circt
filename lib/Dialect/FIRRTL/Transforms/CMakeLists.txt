add_circt_dialect_library(CIRCTFIRRTLTransforms
  BlackBoxReader.cpp
  BlackboxMemory.cpp
  CheckWidths.cpp
  ExpandWhens.cpp
  ModuleInliner.cpp
  IMConstProp.cpp
  InferWidths.cpp
  LowerTypes.cpp

  DEPENDS
  CIRCTFIRRTLTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  CIRCTSV
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
