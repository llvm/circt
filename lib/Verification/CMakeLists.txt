add_circt_library(CIRCTSMT
  Solver.cpp
  Circuit.cpp
  LogicExporter.cpp


  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTCalyx
  CIRCTESI
  CIRCTFIRRTL
  CIRCTFSM
  CIRCTHandshake
  CIRCTHWArith
  CIRCTInteropDialect
  CIRCTLLHD
  CIRCTMoore
  CIRCTPipelineOps
  CIRCTSSP
  CIRCTSystemC

)
