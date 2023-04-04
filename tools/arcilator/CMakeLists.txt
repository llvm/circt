set(LLVM_LINK_COMPONENTS Support)

add_llvm_tool(arcilator arcilator.cpp)
target_link_libraries(arcilator
  PRIVATE
  CIRCTArc
  CIRCTArcToLLVM
  CIRCTArcTransforms
  CIRCTCombToArith
  CIRCTConvertToArcs
  CIRCTSupport
  MLIRParser
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)

llvm_update_compile_flags(arcilator)
mlir_check_all_link_libraries(arcilator)
