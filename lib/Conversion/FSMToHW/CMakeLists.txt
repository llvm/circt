add_circt_conversion_library(CIRCTFSMToHW
    FSMToHW.cpp

    DEPENDS
    CIRCTConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CIRCTComb
    CIRCTHW
    CIRCTFSM
    CIRCTSeq
    CIRCTSV
    CIRCTSupport
    MLIRTransforms
)
