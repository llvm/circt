set(HANDSHAKE_SRCS
  HandshakeDialect.cpp
  HandshakeExecutableOps.cpp
  HandshakeOps.cpp
  HandshakeUtils.cpp
  )

add_circt_dialect_library(CIRCTHandshake
  ${HANDSHAKE_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  CIRCTSeq
  CIRCTESI
  MLIRFuncDialect
  MLIRIR

  DEPENDS
  MLIRArithDialect
  MLIRHandshakeInterfacesIncGen
  MLIRHandshakeCanonicalizationIncGen
  MLIRHandshakeAttributesIncGen
  MLIRHandshakeEnumsIncGen
  )

add_subdirectory(Transforms)
