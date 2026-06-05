add_mlir_dialect_library(CIRCTStandalone
  CIRCTStandaloneDialect.cpp
  CIRCTStandaloneOps.cpp
  CIRCTStandalonePasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_STANDALONE_SOURCE_DIR}/include/CIRCTStandalone

  DEPENDS
  MLIRCIRCTStandaloneOpsIncGen
  CIRCTStandalonePassesIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
)
