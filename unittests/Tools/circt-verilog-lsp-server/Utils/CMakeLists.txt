add_circt_unittest(CIRCTVerilogLSPTests
  PendingChangesTest.cpp
)

target_link_libraries(CIRCTVerilogLSPTests
  PRIVATE
  CIRCTVerilogLspServerUtils
  CIRCTVerilogLspUtilsHeaders
)
