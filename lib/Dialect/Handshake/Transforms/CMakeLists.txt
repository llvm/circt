add_circt_dialect_library(CIRCTHandshakeTransforms
  Analysis.cpp
  Buffers.cpp
  LegalizeMemrefs.cpp
  LockFunctions.cpp
  LowerExtmemToHW.cpp
  Materialization.cpp
  SplitMerges.cpp

  DEPENDS
  CIRCTHandshakeTransformsIncGen
  MLIRArithDialect

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTESI
  CIRCTHandshake
  CIRCTSupport
  CIRCTTransforms
  MLIRIR
  MLIRSCFDialect
  MLIRPass
  MLIRTransformUtils
  MLIRMemRefDialect
  MLIRAffineDialect
  )
