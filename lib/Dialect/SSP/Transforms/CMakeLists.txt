add_circt_dialect_library(CIRCTSSPTransforms
  Print.cpp
  Roundtrip.cpp
  Schedule.cpp

  DEPENDS
  CIRCTSSPTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTScheduling
  CIRCTSSP
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
