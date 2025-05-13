add_circt_dialect_library(CIRCTAIG
  AIGDialect.cpp
  AIGOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/AIG

  LINK_LIBS PUBLIC
  MLIRIR
  CIRCTHW
  CIRCTComb

  DEPENDS
  MLIRAIGIncGen
)

add_subdirectory(Analysis)
add_subdirectory(Transforms)
