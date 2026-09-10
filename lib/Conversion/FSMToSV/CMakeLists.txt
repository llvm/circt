add_circt_conversion_library(CIRCTFSMToSV
    FSMToSV.cpp

    DEPENDS
    CIRCTConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CIRCTComb
    CIRCTHW
    CIRCTEmit
    CIRCTFSM
    CIRCTSeq
    CIRCTSV
    CIRCTSupport
    MLIRTransforms
)
