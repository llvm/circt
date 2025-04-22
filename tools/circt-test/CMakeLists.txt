set(libs
  CIRCTComb
  CIRCTExportVerilog
  CIRCTHW
  CIRCTOM
  CIRCTSeq
  CIRCTSeqToSV
  CIRCTSim
  CIRCTSimToSV
  CIRCTSV
  CIRCTSVTransforms
  CIRCTVerif
  CIRCTVerifToSV
  CIRCTVerifTransforms

  MLIRLLVMDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSCFDialect

  MLIRBytecodeReader
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_circt_tool(circt-test circt-test.cpp DEPENDS ${libs})
target_link_libraries(circt-test PRIVATE ${libs})

llvm_update_compile_flags(circt-test)
mlir_check_all_link_libraries(circt-test)

set(all_runner_paths "")
foreach(runner sby circt-bmc)
  set(src_path ${CMAKE_CURRENT_SOURCE_DIR}/circt-test-runner-${runner}.py)
  set(dst_path ${CIRCT_TOOLS_DIR}/circt-test-runner-${runner}.py)
  add_custom_command(
    OUTPUT ${dst_path}
    DEPENDS ${src_path}
    COMMAND ${CMAKE_COMMAND} -E copy ${src_path} ${dst_path}
  )
  list(APPEND all_runner_paths ${dst_path})
endforeach()
add_custom_target(circt-test-runners DEPENDS ${all_runner_paths})
add_dependencies(circt-test circt-test-runners)
