add_circt_dialect_library(CIRCTComb
  CombFolds.cpp
  CombOps.cpp
  CombAnalysis.cpp
  CombDialect.cpp
  InferIntRangeInterfaceImpls.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Comb

  DEPENDS
  CIRCTHW
  MLIRCombIncGen
  MLIRCombEnumsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRInferIntRangeInterface
   )

add_dependencies(circt-headers MLIRCombIncGen MLIRCombEnumsIncGen)

add_subdirectory(Transforms)
