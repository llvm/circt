set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(hir-builder
 hir-builder.cpp
 MLIRGen.cpp
)

llvm_update_compile_flags(hir-builder)
target_link_libraries(hir-builder
  PRIVATE
  CIRCTHIR
  CIRCTHIRTransforms

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandard
  MLIRTransforms
  MLIRLLVMIR
  )
