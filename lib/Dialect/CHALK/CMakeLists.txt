add_circt_dialect_library(CIRCTCHALK
  CHALKDialect.cpp
  CHALKOps.cpp
  CHALKTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/CHALK

  DEPENDS
  MLIRCHALKIncGen

  LINK_LIBS PUBLIC
  CIRCTSupport
  MLIRIR
  MLIRPass
  )

add_subdirectory(Transforms)
