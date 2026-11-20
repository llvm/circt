add_circt_dialect_library(CIRCTDebug
  DebugDialect.cpp
  DebugOps.cpp
  DebugTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Debug

  DEPENDS
  MLIRDebugIncGen
  CIRCTDebugEnumsIncGen
  CIRCTDebugAttributesIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
