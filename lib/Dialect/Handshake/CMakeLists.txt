set(srcs
  HandshakeOps.cpp
  HandshakeDialect.cpp
  )


set(LLVM_OPTIONAL_SOURCES
  ${srcs}
  DialectRegistration.cpp
  mlir_std_runner.cpp
  )

add_circt_dialect_library(CIRCTHandshake
  ${srcs}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRIR

  DEPENDS
  MLIRHandshakeInterfacesIncGen
  MLIRHandshakeCanonicalizationIncGen
  )
