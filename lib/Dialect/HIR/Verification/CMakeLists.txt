add_mlir_library(MLIRScheduleVerifier
  ScheduleVerifier.cpp

  LINK_LIBS PUBLIC
  MLIRHIR
  )
