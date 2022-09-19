set(LLVM_OPTIONAL_SOURCES
  ControlFlowLoopAnalysis.cpp
  DependenceAnalysis.cpp
  SchedulingAnalysis.cpp
  SignalTracingAnalysis.cpp
  TestPasses.cpp
  )

add_circt_library(CIRCTControlFlowLoopAnalysis
  ControlFlowLoopAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  )

add_circt_library(CIRCTDependenceAnalysis
  DependenceAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineUtils
  MLIRTransformUtils
  )

add_circt_library(CIRCTSchedulingAnalysis
  SchedulingAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRIR
  CIRCTControlFlowLoopAnalysis
  CIRCTDependenceAnalysis
  CIRCTScheduling
  )

add_circt_library(CIRCTSignalTracingAnalysis
  SignalTracingAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAnalysis
  CIRCTHW
  )

add_circt_library(CIRCTAnalysisTestPasses
  TestPasses.cpp

  LINK_LIBS PUBLIC
  CIRCTControlFlowLoopAnalysis
  CIRCTDependenceAnalysis
  CIRCTSchedulingAnalysis
  CIRCTSignalTracingAnalysis
  CIRCTHW
  MLIRPass
  )
