add_circt_dialect_library(CIRCTHIRAnalysis

  FanoutAnalysis.cpp
  ScheduleAnalysis.cpp


  LINK_LIBS PUBLIC
  MLIRTransformUtils
  CIRCTHIR
  )
