add_circt_conversion_library(CIRCTCoreToFSM
    CoreToFSM.cpp

    DEPENDS
    CIRCTConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CIRCTComb
    CIRCTEmit
    CIRCTHW
    CIRCTFSM
    CIRCTSeq
    CIRCTSV
    CIRCTSupport
    MLIRTransforms
)
