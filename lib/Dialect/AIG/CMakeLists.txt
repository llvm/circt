add_circt_dialect_library(CIRCTAIG
  AIGOps.cpp
  AIGDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/AIG

  LINK_LIBS PUBLIC
  MLIRIR
  CIRCTHW

  DEPENDS
  CIRCTHW
  MLIRAIGIncGen
)

add_subdirectory(Transforms)