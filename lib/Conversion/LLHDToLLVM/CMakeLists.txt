add_mlir_conversion_library(MLIRLLHDToLLVM 
    LLHDToLLVM.cpp

    DEPENDS
    MLIRLLHDOpsIncGen
    MLIRLLHDConversionPassIncGen
)

set(LIBS
  MLIRLLVMIR
  MLIRStandardToLLVM
  MLIRVector
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )

add_dependencies(MLIRLLHDToLLVM ${LIBS})
target_link_libraries(MLIRLLHDToLLVM
  PUBLIC
  ${LIBS})
