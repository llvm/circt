# circt-lec builds if a logic backend is found
if(CIRCT_LEC_ENABLED)
  list(APPEND CIRCT_LEC_SOURCES
    circt-lec.cpp
    Circuit.cpp
    LogicExporter.cpp
    Solver.cpp
  )

  list(APPEND CIRCT_LEC_LIBS
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
  )

  add_llvm_tool(circt-lec ${CIRCT_LEC_SOURCES})
  target_link_libraries(circt-lec PRIVATE ${CIRCT_LEC_LIBS})
  target_include_directories(circt-lec PRIVATE ${CIRCT_LEC_INCLUDES})

  llvm_update_compile_flags(circt-lec)
  mlir_check_all_link_libraries(circt-lec)
endif()
