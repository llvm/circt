add_circt_dialect_library(CIRCTFIRRTLTransforms
  BlackboxMemory.cpp
  IMConstProp.cpp
  LowerTypes.cpp

  DEPENDS
  CIRCTFIRRTLTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
