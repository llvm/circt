add_circt_conversion_library(CIRCTFSMToSMT
    FSMToSMT.cpp

    DEPENDS
    CIRCTConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CIRCTComb
    CIRCTHW
    CIRCTEmit
    CIRCTFSM
    CIRCTSMT
    CIRCTSupport
    MLIRTransforms
)
