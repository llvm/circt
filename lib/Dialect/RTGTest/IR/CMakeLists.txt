add_circt_dialect_library(CIRCTRTGTestDialect
  RTGTestDialect.cpp
  RTGTestOps.cpp
  RTGTestTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/RTGTest/IR

  DEPENDS
  MLIRRTGTestIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  CIRCTRTGDialect
)
