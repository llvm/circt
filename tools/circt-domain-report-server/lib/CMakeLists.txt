add_circt_library(CIRCTDomainReport
  Report.cpp
  LINK_COMPONENTS Support
)

# GCC versions before 9 keep std::filesystem in a separate library.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(CIRCTDomainReport PRIVATE stdc++fs)
endif()

target_include_directories(CIRCTDomainReport PRIVATE
  ${CIRCT_SOURCE_DIR}/tools/circt-domain-report-server)
