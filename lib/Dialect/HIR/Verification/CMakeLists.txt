add_mlir_library(MLIRHIRScheduleVerifier
  ScheduleVerifier.cpp

  LINK_LIBS PUBLIC
  MLIRHIR
  )
