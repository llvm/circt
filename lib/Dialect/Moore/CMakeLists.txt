add_circt_dialect_library(CIRCTMoore
  MooreAttributes.cpp
  MooreDialect.cpp
  MooreOps.cpp
  MooreTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Moore

  DEPENDS
  MLIRMooreIncGen
  CIRCTMooreAttributesIncGen
  CIRCTMooreEnumsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRMemorySlotInterfaces
)

add_dependencies(circt-headers MLIRMooreIncGen)

add_subdirectory(Transforms)
