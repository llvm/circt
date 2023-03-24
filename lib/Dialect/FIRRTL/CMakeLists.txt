include_directories(.)
add_circt_dialect_library(CIRCTFIRRTL
  CHIRRTLDialect.cpp
  FIRRTLAnnotationHelper.cpp
  FIRRTLAnnotations.cpp
  FIRRTLAttributes.cpp
  FIRRTLDialect.cpp
  FIRRTLFieldSource.cpp
  FIRRTLFolds.cpp
  FIRRTLInstanceGraph.cpp
  FIRRTLOpInterfaces.cpp
  FIRRTLOps.cpp
  FIRRTLTypes.cpp
  FIRRTLUtils.cpp
  NLATable.cpp

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
  CIRCTSupport
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
