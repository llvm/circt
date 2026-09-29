add_circt_unittest(CIRCTDomainReportTests ReportTest.cpp)
target_include_directories(CIRCTDomainReportTests PRIVATE
  ${CIRCT_SOURCE_DIR}/tools/circt-domain-report-server)
target_link_libraries(CIRCTDomainReportTests PRIVATE CIRCTDomainReport)
