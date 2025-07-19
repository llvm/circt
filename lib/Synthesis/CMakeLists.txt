add_circt_library(CIRCTSynthesis
  CutRewriter.cpp

  LINK_LIBS PUBLIC
  CIRCTSupport

  MLIRIR
  MLIRSupport
  MLIRTransforms
)

add_subdirectory(Transforms)
