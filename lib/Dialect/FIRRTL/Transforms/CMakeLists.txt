add_circt_dialect_library(CIRCTFIRRTLTransforms
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
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
