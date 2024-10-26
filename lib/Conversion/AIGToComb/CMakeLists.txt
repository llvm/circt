add_circt_conversion_library(CIRCTAIGToComb
  AIGToComb.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Conversion/AIGToComb

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  CIRCTAIG
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
)
