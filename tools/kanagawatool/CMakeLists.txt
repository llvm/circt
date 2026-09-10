set(LLVM_LINK_COMPONENTS
    Support
)

add_circt_tool(kanagawatool
    kanagawatool.cpp
)
llvm_update_compile_flags(kanagawatool)
target_link_libraries(kanagawatool
    PRIVATE

    CIRCTExportVerilog
    CIRCTHandshake
    CIRCTHandshakeToHW
    CIRCTHandshakeTransforms
    CIRCTHW
    CIRCTHWTransforms
    CIRCTSeq
    CIRCTSeqToSV
    CIRCTSeqTransforms
    CIRCTCFToHandshake
    CIRCTSV
    CIRCTSVTransforms
    CIRCTKanagawa
    CIRCTKanagawaTransforms
    CIRCTCombTransforms
    CIRCTHWToSV
    CIRCTTransforms
    CIRCTPipelineOps
    CIRCTPipelineToHW
    CIRCTDCToHW
    CIRCTHandshakeToDC

    CIRCTPipelineTransforms
    CIRCTDC
    CIRCTDCTransforms
    CIRCTESI

    MLIRIR
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIROptLib
    MLIRParser
    MLIRControlFlowDialect
    MLIRSupport
    MLIRTransforms
    MLIRSCFToControlFlow
)
