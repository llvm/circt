add_subdirectory(IR)
add_subdirectory(Transforms)

add_circt_library(CIRCTRTGReductions
  RTGReductions.cpp

  LINK_LIBS PUBLIC
  CIRCTReduceLib
  CIRCTRTGDialect
  MLIRIR
)
