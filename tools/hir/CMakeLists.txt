set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(hir-builder
  mlir/HIRGen.cpp
  mlir/EmitExpr.cpp
  #apps/polymageHelpers.cpp
  hir-builder.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/lib/Dialect/HIR
)

llvm_update_compile_flags(hir-builder)
target_link_libraries(hir-builder
  PRIVATE
  CIRCTHIR

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandard
  MLIRTransforms
  MLIRLLVMIR
  )
