add_circt_dialect_library(CIRCTSMT
  SMTAttributes.cpp
  SMTDialect.cpp
  SMTOps.cpp
  SMTTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/SMT

  DEPENDS
  CIRCTSMTAttrIncGen
  CIRCTSMTEnumsIncGen
  MLIRSMTIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTSupport
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
)

add_dependencies(circt-headers
  MLIRSMTIncGen
)
