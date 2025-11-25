add_circt_dialect_library(CIRCTCombTransforms
  LowerComb.cpp
  IntRangeOptimizations.cpp
  IntRangeAnnotations.cpp
  BalanceMux.cpp
  AssumeTwoValued.cpp

  DEPENDS
  CIRCTCombTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSV
  CIRCTComb
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
