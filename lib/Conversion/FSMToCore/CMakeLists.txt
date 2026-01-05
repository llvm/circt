add_circt_conversion_library(CIRCTFSMToCore
    FSMToCore.cpp

    DEPENDS
    CIRCTConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CIRCTComb
    CIRCTFSM
    CIRCTHW
    CIRCTSeq
    CIRCTSupport
    MLIRTransforms
)
