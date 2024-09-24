add_circt_dialect_library(CIRCTRTGDialect
  RTGDialect.cpp
  RTGOps.cpp
  RTGTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/RTG/IR

  DEPENDS
  MLIRRTGIncGen
  CIRCTRTGInterfacesIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBytecodeOpInterface
)
