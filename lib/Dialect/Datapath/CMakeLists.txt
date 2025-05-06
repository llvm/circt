add_circt_dialect_library(CIRCTDatapath
  DatapathDialect.cpp
  DatapathOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Datapath

  DEPENDS
  CIRCTComb
  # MLIRCombIncGen
  # MLIRCombEnumsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  MLIRIR
  MLIRInferTypeOpInterface
  )

# add_dependencies(circt-headers)
