add_circt_unittest(CIRCTSupportTests
  FVIntTest.cpp
  JSONTest.cpp
  PrettyPrinterTest.cpp
  SATSolverTest.cpp
  SparseOpSCCTest.cpp
  TruthTableTest.cpp
)

target_link_libraries(CIRCTSupportTests
  PRIVATE
  CIRCTSupport
  CIRCTHW
  CIRCTComb
  CIRCTSeq
  MLIRIR
  MLIRParser
)
