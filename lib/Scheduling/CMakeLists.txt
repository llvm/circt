add_circt_library(CIRCTScheduling
  ASAPScheduler.cpp
  Problems.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
  MLIRSupport
  )
