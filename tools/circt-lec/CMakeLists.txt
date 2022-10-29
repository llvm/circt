# circt-lec builds if a logic backend is found
if(CIRCT_LEC_ENABLED)
  add_llvm_tool(circt-lec
    circt-lec.cpp
    Circuit.cpp
    LogicExporter.cpp
    Solver.cpp
  )

  llvm_update_compile_flags(circt-lec)
  target_link_libraries(circt-lec
    PRIVATE
    CIRCTCalyx
    CIRCTESI
    CIRCTFIRRTL
    CIRCTFSM
    CIRCTHandshake
    CIRCTHWArith
    CIRCTInteropDialect
    CIRCTLLHD
    CIRCTMoore
    CIRCTPipelineOps
    CIRCTSSP
    CIRCTSystemC

    ${Z3_LIBRARIES}
  )
endif()
