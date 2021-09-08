file(GLOB globbed *.cpp)

add_circt_dialect_library(CIRCTFSM
  ${globbed}

  DEPENDS
  MLIRFSMIncGen

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRIR
  )
