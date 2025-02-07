add_circt_library(CIRCTRtgToolLib
  RtgToolOptions.cpp

  LINK_LIBS PUBLIC
  CIRCTRTGDialect
  CIRCTRTGTransforms
  CIRCTSupport

  MLIRIR
  MLIRPass
  MLIRTransforms
)
