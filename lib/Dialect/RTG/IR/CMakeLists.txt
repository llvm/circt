add_circt_dialect_library(CIRCTRTGDialect
  RTGOps.cpp
  RTGDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/RTG/IR

  DEPENDS
  MLIRRTGIncGen
  MLIRRTGEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
