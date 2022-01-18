add_circt_dialect_library(CIRCTGAA
  GAADialect.cpp
  GAAOps.cpp
  GAAOpInterfaces.cpp
  InstanceGraph.cpp
  CallInfo.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/GAA

  DEPENDS
  CIRCTHW
  MLIRGAAIncGen
  CIRCTGAAAttrsIncGen
  CIRCTGAAOpInterfacesIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTSupport
  CIRCTHW
  MLIRIR
  )
add_subdirectory(Transforms)
