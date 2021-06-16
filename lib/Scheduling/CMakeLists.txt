add_circt_library(CIRCTScheduling
  ASAPScheduler.cpp
  Scheduler.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
  MLIRSupport
  )
