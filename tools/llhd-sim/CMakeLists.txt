get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        CIRCTLLHD
        CIRCTComb
        CIRCTLLHDToLLVM
        CIRCTLLHDSimEngine
        )

# llhd-sim fails to link on Windows with MSVC.
IF(CIRCT_LLHD_SIM_ENABLED)
  add_llvm_executable(llhd-sim
    llhd-sim.cpp)
  
  llvm_update_compile_flags(llhd-sim)
  target_link_libraries(llhd-sim PRIVATE ${LIBS})
endif()
