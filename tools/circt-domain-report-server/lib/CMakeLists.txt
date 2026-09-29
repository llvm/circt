add_circt_library(CIRCTDomainReport
  Report.cpp
  LINK_COMPONENTS Support
)
target_include_directories(CIRCTDomainReport PRIVATE
  ${CIRCT_SOURCE_DIR}/tools/circt-domain-report-server)
