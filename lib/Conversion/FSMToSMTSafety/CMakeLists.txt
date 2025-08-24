add_circt_conversion_library(CIRCTFSMToSMTSafety
    FSMToSMTSafety.cpp

    DEPENDS
    CIRCTConversionPassIncGen

    LINK_COMPONENTS
    Core


    LINK_LIBS PUBLIC
    MLIRSMT

    CIRCTComb
    CIRCTVerif
    CIRCTHW
    CIRCTEmit
    CIRCTFSM
    CIRCTSupport
    MLIRTransforms
)