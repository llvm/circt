set(LLVM_LINK_COMPONENTS
    Support
)

add_circt_tool(ibistool
    ibistool.cpp
)
llvm_update_compile_flags(ibistool)
target_link_libraries(ibistool
    PRIVATE

    CIRCTExportChiselInterface
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
    CIRCTIbis
    CIRCTIbisTransforms
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
