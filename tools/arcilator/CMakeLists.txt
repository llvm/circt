if(ARCILATOR_JIT_ENABLED)
  add_compile_definitions(ARCILATOR_ENABLE_JIT)
  add_subdirectory(jit-env)
  set(ARCILATOR_JIT_LLVM_COMPONENTS native)
  set(ARCILATOR_JIT_DEPS MLIRExecutionEngine arc-jit-env)
endif()

set(LLVM_LINK_COMPONENTS Support ${ARCILATOR_JIT_LLVM_COMPONENTS})

add_circt_tool(arcilator arcilator.cpp)
target_link_libraries(arcilator
  PRIVATE
  CIRCTArc
  CIRCTHWTransforms
  CIRCTArcToLLVM
  CIRCTArcTransforms
  CIRCTCombToArith
  CIRCTConvertToArcs
  CIRCTEmit
  CIRCTExportArc
  CIRCTOM
  CIRCTSeqToSV
  CIRCTSeqTransforms
  CIRCTSimTransforms
  CIRCTSupport
  CIRCTTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRDLTIDialect
  MLIRFuncInlinerExtension
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRParser
  MLIRTargetLLVMIRExport

  ${ARCILATOR_JIT_DEPS}
)

llvm_update_compile_flags(arcilator)
mlir_check_all_link_libraries(arcilator)

configure_file(arcilator-header-cpp.py
  ${CIRCT_TOOLS_DIR}/arcilator-header-cpp.py)
add_custom_target(arcilator-header-cpp SOURCES
  ${CIRCT_TOOLS_DIR}/arcilator-header-cpp.py)

configure_file(arcilator-runtime.h
  ${CIRCT_TOOLS_DIR}/arcilator-runtime.h)
add_custom_target(arcilator-runtime-header SOURCES
  ${CIRCT_TOOLS_DIR}/arcilator-runtime.h)

if(ARCILATOR_JIT_ENABLED)
  target_include_directories(arcilator PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/jit-env>
  )
endif()
