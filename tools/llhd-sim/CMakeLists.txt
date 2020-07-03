get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRLLHD
        MLIRLLHDToLLVM
        CIRCTLLHDSimEngine
        )

add_llvm_executable(llhd-sim
 llhd-sim.cpp)
 
llvm_update_compile_flags(llhd-sim)
target_link_libraries(llhd-sim PRIVATE ${LIBS})
