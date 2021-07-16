file(GLOB globbed *.cpp)

add_mlir_dialect_library(CIRCTFSM
  ${globbed}

  DEPENDS
  MLIRFSMIncGen
  )
