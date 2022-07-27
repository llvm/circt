add_circt_dialect_library(CIRCTSystemC
  SystemCAttributes.cpp
  SystemCDialect.cpp
  SystemCTypes.cpp
  SystemCOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/SystemC

  DEPENDS
  MLIRSystemCIncGen
  CIRCTSystemCAttributesIncGen
  CIRCTSystemCEnumsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
)

add_dependencies(circt-headers MLIRSystemCIncGen)
