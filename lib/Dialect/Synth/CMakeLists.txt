##===----------------------------------------------------------------------===//
##
## Implementation files for the Synth dialect.
##
##===----------------------------------------------------------------------===//

add_circt_dialect_library(CIRCTSynth
  SynthDialect.cpp
  SynthOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Synth

  DEPENDS
  MLIRSynthIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
)

add_subdirectory(Transforms)
