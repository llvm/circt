add_circt_library(CIRCTScheduling
  Scheduler.cpp
  ASAPScheduler.cpp

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRIR
  )
