set(LLVM_OPTIONAL_SOURCES
    State.cpp
    Engine.cpp
    signals-runtime-wrappers.cpp
    Trace.cpp
)

add_mlir_library(CIRCTLLHDSimState
    State.cpp
)

add_mlir_library(CIRCTLLHDSimTrace
    Trace.cpp

    LINK_LIBS PUBLIC
    CIRCTLLHDSimState
)

add_mlir_library(circt-llhd-signals-runtime-wrappers SHARED
    signals-runtime-wrappers.cpp

    LINK_LIBS PUBLIC
    CIRCTLLHDSimState
)

add_mlir_library(CIRCTLLHDSimEngine
    Engine.cpp

    LINK_LIBS PUBLIC
    MLIRLLHD
    MLIRLLHDToLLVM
    CIRCTLLHDSimState
    circt-llhd-signals-runtime-wrappers
    MLIRExecutionEngine
    CIRCTLLHDSimTrace
    )
