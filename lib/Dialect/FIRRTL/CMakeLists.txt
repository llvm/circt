add_circt_dialect_library(CIRCTFIRRTL
  FIRRTLAnnotations.cpp
  FIRRTLAttributes.cpp
  FIRRTLDialect.cpp
  FIRRTLFolds.cpp
  FIRRTLOpInterfaces.cpp
  FIRRTLOps.cpp
  FIRRTLTypes.cpp
  InstanceGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/FIRRTL

  DEPENDS
  MLIRFIRRTLIncGen
  CIRCTFIRRTLEnumsIncGen
  CIRCTFIRRTLCanonicalizationIncGen
  CIRCTFIRRTLOpInterfacesIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRIR
  MLIRPass
  )

add_dependencies(circt-headers
  MLIRFIRRTLIncGen
  CIRCTFIRRTLEnumsIncGen
  CIRCTFIRRTLCanonicalizationIncGen
  )

add_subdirectory(Export)
add_subdirectory(Import)
add_subdirectory(Transforms)
