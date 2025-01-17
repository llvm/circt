add_circt_dialect_library(CIRCTRTGDialect
  RTGAttrInterfaces.cpp
  RTGDialect.cpp
  RTGISAAssemblyOpInterfaces.cpp
  RTGOpInterfaces.cpp
  RTGOps.cpp
  RTGTypeInterfaces.cpp
  RTGTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/RTG/IR

  DEPENDS
  CIRCTRTGAttrInterfacesIncGen
  CIRCTRTGEnumsIncGen
  CIRCTRTGISAAssemblyOpInterfacesIncGen
  CIRCTRTGOpInterfacesIncGen
  CIRCTRTGTypeInterfacesIncGen
  MLIRRTGIncGen
    
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBytecodeOpInterface
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
