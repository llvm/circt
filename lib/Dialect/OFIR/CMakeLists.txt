include_directories(.)
add_circt_dialect_library(CIRCTOFIR
  OFIRDialect.cpp
  OFIROps.cpp
  OFIRTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/OFIR

  DEPENDS
  MLIROFIRIncGen
  CIRCTOFIREnumsIncGen
  CIRCTOFIRCanonicalizationIncGen

  LINK_COMPONENTS

  Support

  LINK_LIBS PUBLIC
  CIRCTSupport
  CIRCTFIRRTL
  CIRCTHW
  MLIRIR
  MLIRPass
  )

add_dependencies(circt-headers
  MLIROFIRIncGen
  CIRCTOFIREnumsIncGen
  )

