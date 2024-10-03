set(LLVM_OPTIONAL_SOURCES
  DebugAnalysis.cpp
  DebugInfo.cpp
  DependenceAnalysis.cpp
  FIRRTLInstanceInfo.cpp
  OpCountAnalysis.cpp
  SchedulingAnalysis.cpp
  TestPasses.cpp
)

add_circt_library(CIRCTDebugAnalysis
  DebugAnalysis.cpp
  DebugInfo.cpp

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTDebug
  CIRCTHW
  MLIRIR
)

add_circt_library(CIRCTDependenceAnalysis
  DependenceAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineUtils
  MLIRTransformUtils
)

add_circt_library(CIRCTOpCountAnalysis
  OpCountAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)

add_circt_library(CIRCTSchedulingAnalysis
  SchedulingAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRIR
  CIRCTDependenceAnalysis
  CIRCTScheduling
)

add_circt_library(CIRCTFIRRTLAnalysis
  FIRRTLInstanceInfo.cpp

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
)

add_circt_library(CIRCTAnalysisTestPasses
  TestPasses.cpp

  LINK_LIBS PUBLIC
  CIRCTDebugAnalysis
  CIRCTDependenceAnalysis
  CIRCTFIRRTLAnalysis
  CIRCTSchedulingAnalysis
  CIRCTHW
  MLIRPass
)
