add_circt_conversion_library(CIRCTCombToAIG
  CombToAIG.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Conversion/CombToAIG

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  CIRCTAIG
  CIRCTDatapath
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
)
