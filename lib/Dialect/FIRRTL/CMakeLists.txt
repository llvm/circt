set(LLVM_TARGET_DEFINITIONS FIRRTLIntrinsics.td)
circt_tablegen(CIRCT FIRRTLIntrinsics.cpp.inc --firrtl-intrinsics TARGETS MLIRFIRRTLIncGen)
circt_tablegen(CIRCT ${CIRCT_BINARY_DIR}/docs/Dialects/FIRRTLIntrinsics.md.inc --firrtl-intrinsics-doc TARGETS circt-doc)

set(CIRCT_FIRRTL_Sources
  CHIRRTLDialect.cpp
  CHIRRTLTypes.cpp
  FieldRefCache.cpp
  FIRRTLAnnotationHelper.cpp
  FIRRTLAnnotations.cpp
  FIRRTLAttributes.cpp
  FIRRTLDialect.cpp
  FIRRTLFieldSource.cpp
  FIRRTLFolds.cpp
  FIRRTLInstanceGraph.cpp
  FIRRTLInstanceImplementation.cpp
  FIRRTLIntrinsics.cpp
  FIRRTLOpInterfaces.cpp
  FIRRTLOps.cpp
  FIRRTLTypes.cpp
  FIRRTLUtils.cpp
  NLATable.cpp
)

set(LLVM_OPTIONAL_SOURCES
  ${CIRCT_FIRRTL_Sources}
  FIRRTLReductions.cpp
)

add_circt_dialect_library(CIRCTFIRRTL
  ${CIRCT_FIRRTL_Sources}

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
  CIRCTOM
  CIRCTSeq
  CIRCTSV
  MLIRIR
  MLIRPass
)

target_include_directories(CIRCTFIRRTL PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(CIRCTFIRRTL PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_circt_library(CIRCTFIRRTLReductions
  FIRRTLReductions.cpp

  LINK_LIBS PUBLIC
  CIRCTReduceLib
  CIRCTFIRRTL
  CIRCTFIRRTLTransforms
  MLIRIR
)

add_dependencies(circt-headers
  MLIRFIRRTLIncGen
  CIRCTFIRRTLEnumsIncGen
  CIRCTFIRRTLCanonicalizationIncGen
)

add_subdirectory(Export)
add_subdirectory(Import)
add_subdirectory(Transforms)
