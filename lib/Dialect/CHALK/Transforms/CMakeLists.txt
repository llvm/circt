add_circt_dialect_library(CIRCTCHALKTransforms
    InitPlacement.cpp

    DEPENDS
    CIRCTCHALKTransformsIncGen

    LINK_LIBS PUBLIC
    CIRCTSupport
    MLIRIR
    MLIRPass
    MLIRTransformUtils
)
