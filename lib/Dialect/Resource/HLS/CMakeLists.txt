add_circt_dialect_library(CIRCTHLS
        HLSDialect.cpp
        HLSOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Resource/HLS

        DEPENDS
        CIRCTHLSIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRMemRefDialect
        MLIRSideEffectInterfaces
)
