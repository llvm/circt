file(GLOB globbed *.cpp)
add_circt_dialect_library(CIRCTSeq
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Seq

  DEPENDS
  CIRCTRTL
  MLIRSeqIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTRTL
  MLIRIR
   )

add_dependencies(circt-headers MLIRSeqIncGen)
