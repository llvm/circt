set(LLVM_OPTIONAL_SOURCES
  DialectRegistration.cpp
  HandshakeOps.cpp
  mlir_std_runner.cpp
  )

add_mlir_dialect_library(MLIRHandshakeOps
  HandshakeOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRStandardOps
  MLIRIR

  DEPENDS
  MLIRHandshakeOpsIncGen
  MLIRHandshakeInterfacesIncGen
  MLIRHandshakeRewritersIncGen
  )
