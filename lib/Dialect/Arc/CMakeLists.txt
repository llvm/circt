set(srcs
  ArcDialect.cpp
  ArcFolds.cpp
  ArcOps.cpp
  ArcTypes.cpp
)

set(LLVM_OPTIONAL_SOURCES
  ${srcs}
  ArcReductions.cpp
)

add_circt_dialect_library(CIRCTArc
  ${srcs}

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

add_circt_library(CIRCTArcReductions
  ArcReductions.cpp

  LINK_LIBS PUBLIC
  CIRCTReduceLib
  CIRCTArc
  CIRCTArcTransforms
  CIRCTConvertToArcs
  MLIRIR
)

add_dependencies(circt-headers
  MLIRArcIncGen
)

add_subdirectory(Interfaces)
add_subdirectory(Transforms)
