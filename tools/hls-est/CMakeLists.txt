add_circt_tool(hls-est main.cpp)

target_link_libraries(hls-est PRIVATE
        # Upstream dialects
        MLIRAffineDialect
        MLIRArithDialect
        MLIRMemRefDialect
        MLIRSCFDialect
        MLIRFuncDialect
        MLIRControlFlowDialect
        MLIRIndexDialect
        MLIRTransforms

        # CIRCT dialects
        CIRCTLoopSchedule
        CIRCTHW
        CIRCTComb
        CIRCTSeq
        CIRCTCalyx

        # Your libraries
        HLSInterfaces
        HLSPasses
        CIRCTHLS
        

        # Driver infrastructure
        MLIROptLib
)