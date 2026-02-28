add_circt_conversion_library(CIRCTCombToDatapath
  CombToDatapath.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Conversion/CombToDatapath

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  CIRCTDatapath
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
)
