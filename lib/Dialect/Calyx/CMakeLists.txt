##===- CMakeLists.txt - Calyx dialect code --------------------*- cmake -*-===//
##
## Implementation files the Calyx dialect.
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
        )



add_circt_dialect_library(CIRCTCalyx
        ${srcs}

        DEPENDS
        MLIRCalyxEnumsIncGen
        ${Calyx_Deps}

        LINK_COMPONENTS
        Core
        Support

        LINK_LIBS PUBLIC
        ${Calyx_LinkLibs}
        )
