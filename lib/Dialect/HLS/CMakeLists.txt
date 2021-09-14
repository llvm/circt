file(GLOB globbed *.cpp)

add_circt_dialect_library(CIRCTHLS
  ${globbed}

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  CIRCTHLSAttributesIncGen
  CIRCTHLSEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

add_dependencies(circt-headers CIRCTHLSAttributesIncGen CIRCTHLSEnumsIncGen)

