add_circt_dialect_library(CIRCTHandshakeTransforms
  Analysis.cpp
  PassHelpers.cpp
  Materialization.cpp
  Buffers.cpp
  LockFunctions.cpp
  LowerExtmemToHW.cpp

  DEPENDS
  CIRCTHandshakeTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHandshake
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRMemRefDialect
  MLIRAffineDialect
  )
