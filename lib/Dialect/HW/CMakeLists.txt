add_circt_dialect_library(
  CIRCTHW
  HWAttributes.cpp
  HWDialect.cpp
  HWInstanceGraph.cpp
  HWOpInterfaces.cpp
  HWOps.cpp
  HWTypes.cpp
  InstanceGraphBase.cpp
  ModuleImplementation.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/HW
  
  DEPENDS
  MLIRHWIncGen
  MLIRHWAttrIncGen
  MLIRHWEnumsIncGen
  
  LINK_COMPONENTS
  Support
  
  LINK_LIBS PUBLIC
  MLIRIR
)

add_subdirectory(Transforms)
