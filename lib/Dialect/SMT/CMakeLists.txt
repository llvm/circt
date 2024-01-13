add_circt_dialect_library(CIRCTSMT
  SMTDialect.cpp
  SMTOps.cpp
  SMTTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/SMT

  DEPENDS
  CIRCTSMTEnumsIncGen
  MLIRSMTIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

add_dependencies(circt-headers
  MLIRSMTIncGen
)
