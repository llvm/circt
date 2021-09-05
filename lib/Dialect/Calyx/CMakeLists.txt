##===- CMakeLists.txt - Calyx dialect code --------------------*- cmake -*-===//
##
## Implementation files for the Calyx dialect.
##
##===----------------------------------------------------------------------===//

set(srcs
  CalyxDialect.cpp
  CalyxOps.cpp
  )

set(Calyx_LinkLibs
  CIRCTSupport
  CIRCTComb
  CIRCTSV
  CIRCTHW
  MLIRIR
  MLIRTransforms
  MLIRTranslation
  )

set(Calyx_Deps
  ${Calyx_LinkLibs}
  MLIRCalyxEnumsIncGen
  MLIRCalyxAttrsIncGen
  MLIRCalyxInterfacesIncGen
  )

add_circt_dialect_library(CIRCTCalyx
  ${srcs}

  DEPENDS
  ${Calyx_Deps}

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC
  ${Calyx_LinkLibs}
  )

add_subdirectory(Export)
add_subdirectory(Transforms)
