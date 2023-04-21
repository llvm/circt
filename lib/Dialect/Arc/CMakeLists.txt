add_circt_dialect_library(CIRCTArc
  ArcDialect.cpp
  ArcFolds.cpp
  ArcOps.cpp
  ArcTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Arc

  DEPENDS
  CIRCTArcEnumsIncGen
  CIRCTArcInterfacesIncGen
  MLIRArcIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

add_dependencies(circt-headers
  MLIRArcIncGen
)

add_subdirectory(Interfaces)
add_subdirectory(Transforms)
