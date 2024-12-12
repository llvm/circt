add_circt_library(CIRCTRtgToolLib
  RtgToolOptions.cpp

  LINK_LIBS PUBLIC
  CIRCTSupport

  MLIRIR
  MLIRPass
  MLIRTransforms
)
