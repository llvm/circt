add_circt_unittest(CIRCTSimTests
  ExportDPIInterfaceTest.cpp
)

target_link_libraries(CIRCTSimTests
  PRIVATE
  CIRCTHW
  CIRCTExportVerilog
  CIRCTMoore
  CIRCTSV
  CIRCTSim
  CIRCTSimTransforms
  CIRCTSimToSV
  MLIRPass
  MLIRLLVMDialect
  MLIRIR
  MLIRParser
)
