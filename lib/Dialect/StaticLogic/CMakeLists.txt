set(LLVM_OPTIONAL_SOURCES
  DialectRegistration.cpp
  mlir_std_runner.cpp
  Ops.cpp
  )

add_mlir_dialect_library(MLIRStaticLogicOps
  Ops.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRIR

  DEPENDS
  MLIRStaticLogicIncGen
)
