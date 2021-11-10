add_circt_dialect_library(CIRCTCHIRRTL
  CHIRRTLDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/CHIRRTL

  DEPENDS
  MLIRCHIRRTLIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  MLIRIR
  MLIRPass
  )

add_dependencies(circt-headers
  MLIRCHIRRTLIncGen
  )
