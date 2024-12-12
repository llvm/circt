add_circt_dialect_library(CIRCTRTGDialect
  RTGDialect.cpp
  RTGISAAssemblyOpInterfaces.cpp
  RTGOpInterfaces.cpp
  RTGOps.cpp
  RTGTypes.cpp
  RTGTypeInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/RTG/IR

  DEPENDS
  MLIRRTGIncGen
  CIRCTRTGOpInterfacesIncGen
  CIRCTRTGISAAssemblyOpInterfacesIncGen
  CIRCTRTGTypeInterfacesIncGen
    
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBytecodeOpInterface
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
