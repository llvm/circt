##===- CMakeLists.txt - Define a support library --------------*- cmake -*-===//
##
##===----------------------------------------------------------------------===//


add_circt_library(CIRCTSupport
  FieldRef.cpp
  LoweringOptions.cpp
  Path.cpp
  APInt.cpp
  ValueMapper.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRIR
  )
