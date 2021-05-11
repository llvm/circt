add_circt_dialect_library(CIRCTScheduling
  Scheduling.cpp

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRIR

  DEPENDS
  MLIRSchedulingIncGen
  MLIRSchedulingAttributesIncGen
  MLIRSchedulingInterfacesIncGen
)
