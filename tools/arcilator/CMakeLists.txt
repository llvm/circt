if(ARCILATOR_JIT_ENABLED)
  add_compile_definitions(ARCILATOR_ENABLE_JIT)
  add_subdirectory(jit-env)
  set(ARCILATOR_JIT_LLVM_COMPONENTS native)
  set(ARCILATOR_JIT_DEPS MLIRExecutionEngine arc-jit-env)
endif()

set(LLVM_LINK_COMPONENTS Support ${ARCILATOR_JIT_LLVM_COMPONENTS})

set(libs
  CIRCTArc
  CIRCTArcToLLVM
  CIRCTArcTransforms
  CIRCTCombToArith
  CIRCTConvertToArcs
  CIRCTEmitTransforms
  CIRCTExportArc
  CIRCTHWTransforms
  CIRCTLLHD
  CIRCTOMTransforms
  CIRCTSeqToSV
  CIRCTSeqTransforms
  CIRCTSimTransforms
  CIRCTSupport
  CIRCTTransforms
  CIRCTVerif
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRControlFlowDialect
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIndexDialect
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRParser
  MLIRSCFDialect
  MLIRTargetLLVMIRExport
)

add_circt_tool(arcilator arcilator.cpp DEPENDS ${libs})
target_link_libraries(arcilator
  PRIVATE
  ${libs}
  ${ARCILATOR_JIT_DEPS}
)

llvm_update_compile_flags(arcilator)
mlir_check_all_link_libraries(arcilator)

configure_file(arcilator-header-cpp.py
  ${CIRCT_TOOLS_DIR}/arcilator-header-cpp.py)

configure_file(arcilator-runtime.h
  ${CIRCT_TOOLS_DIR}/arcilator-runtime.h)

if(CIRCT_BUILD_TOOLS)
  install(PROGRAMS arcilator-header-cpp.py
    DESTINATION "${CIRCT_TOOLS_INSTALL_DIR}"
    COMPONENT arcilator
  )
  install(FILES arcilator-runtime.h
    DESTINATION include
    COMPONENT arcilator
  )
endif()

if(ARCILATOR_JIT_ENABLED)
  target_include_directories(arcilator PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/jit-env>
  )
endif()
