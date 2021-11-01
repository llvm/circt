##===- CMakeLists.txt - Define a support library --------------*- cmake -*-===//
##
##===----------------------------------------------------------------------===//


add_circt_library(CIRCTSupport
  BackedgeBuilder.cpp
  FieldRef.cpp
  LoweringOptions.cpp
  Path.cpp
  
  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  MLIRIR
  )
