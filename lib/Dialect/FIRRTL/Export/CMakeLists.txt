file(GLOB globbed *.cpp)

add_circt_translation_library(CIRCTExportFIRRTL
  ${globbed}
  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  MLIRTranslation
)
