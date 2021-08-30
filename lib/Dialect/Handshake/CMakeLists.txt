set(HANDSHAKE_SRCS
  HandshakeOps.cpp
  HandshakeDialect.cpp
  )

add_circt_dialect_library(CIRCTHandshake
  ${HANDSHAKE_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRIR

  DEPENDS
  MLIRHandshakeInterfacesIncGen
  MLIRHandshakeCanonicalizationIncGen
  MLIRHandshakeEnumsIncGen
  MLIRHandshakeAttrsIncGen
  )
