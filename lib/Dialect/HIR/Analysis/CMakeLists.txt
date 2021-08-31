add_circt_dialect_library(CIRCTHIRAnalysis

  ScheduleAnalysis.cpp


  LINK_LIBS PUBLIC
  MLIRTransformUtils
  CIRCTHIR
  )
