add_circt_library(CIRCTRtgToolLib
  RtgToolOptions.cpp

  LINK_LIBS PUBLIC
  CIRCTRTGTransforms
  CIRCTSupport

  MLIRIR
  MLIRPass
  MLIRTransforms
)
