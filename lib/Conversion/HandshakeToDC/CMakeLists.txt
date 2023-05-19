add_circt_library(CIRCTHandshakeToDC
    HandshakeToDC.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/HandshakeToDC

    LINK_LIBS PUBLIC
    CIRCTHW
    CIRCTHandshake
    CIRCTHandshakeTransforms
    CIRCTDC
    MLIRIR
    MLIRPass
    MLIRArithDialect
    MLIRFuncDialect
    MLIRSupport
    MLIRTransforms
)
