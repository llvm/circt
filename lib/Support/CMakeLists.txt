##===- CMakeLists.txt - Define a support library --------------*- cmake -*-===//
##
##===----------------------------------------------------------------------===//


file(GLOB globbed *.cpp)

add_circt_library(CIRCTSupport
  ${globbed}
  
  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  MLIRIR
  )

add_subdirectory(Scheduling)
