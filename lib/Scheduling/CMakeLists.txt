add_circt_library(CIRCTScheduling
  Scheduler.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
  MLIRSupport
  )
