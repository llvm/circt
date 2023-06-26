add_circt_library(CIRCTOMEvaluator
  Evaluator.cpp

  DEPENDS
  MLIROMIncGen
  MLIROMAttrIncGen

  LINK_LIBS
  CIRCTOM
  MLIRIR
)
