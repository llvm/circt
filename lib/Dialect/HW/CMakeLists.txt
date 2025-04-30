set(CIRCT_HW_Sources
  ConversionPatterns.cpp
  CustomDirectiveImpl.cpp
  HierPathCache.cpp
  HWAttributes.cpp
  HWEnums.cpp
  HWDialect.cpp
  HWInstanceGraph.cpp
  HWInstanceImplementation.cpp
  HWModuleOpInterface.cpp
  HWOpInterfaces.cpp
  HWOps.cpp
  HWTypeInterfaces.cpp
  HWTypes.cpp
  ModuleImplementation.cpp
  InnerSymbolTable.cpp
  PortConverter.cpp
  InferIntRangeInterfaceImpls.cpp
)

set(LLVM_OPTIONAL_SOURCES
  ${CIRCT_HW_Sources}
  HWReductions.cpp
)

add_circt_dialect_library(CIRCTHW
  ${CIRCT_HW_Sources}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/HW

  DEPENDS
  MLIRHWIncGen
  MLIRHWAttrIncGen
  MLIRHWEnumsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTSupport
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRMemorySlotInterfaces
  MLIRInferIntRangeCommon
  MLIRInferIntRangeInterface
)

add_circt_library(CIRCTHWReductions
  HWReductions.cpp

  LINK_LIBS PUBLIC
  CIRCTReduceLib
  CIRCTHW
  MLIRIR
)

add_subdirectory(Transforms)
