set(LLVM_OPTIONAL_SOURCES
  DialectRegistration.cpp
  mlir_std_runner.cpp
  Ops.cpp
  )

add_circt_dialect_library(CIRCTStaticLogicOps
  StaticLogicAttributes.cpp
  StaticLogicOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRStandard
  MLIRIR

  DEPENDS
  MLIRStaticLogicIncGen
  MLIRStaticLogicAttrIncGen
)
