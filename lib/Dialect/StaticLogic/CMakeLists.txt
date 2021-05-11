set(LLVM_OPTIONAL_SOURCES
  DialectRegistration.cpp
  mlir_std_runner.cpp
  Ops.cpp
  StaticLogicOps.cpp
  StaticLogicPasses.cpp
  )

add_circt_dialect_library(CIRCTStaticLogicOps
  StaticLogicOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRIR

  DEPENDS
  MLIRStaticLogicIncGen
)

add_circt_library(CIRCTStaticLogicPasses
  StaticLogicPasses.cpp

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRIR

  DEPENDS
  MLIRStaticLogicPassesIncGen
)
