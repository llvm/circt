file(GLOB globbed *.cpp)
add_circt_dialect_library(CIRCTFIRRTL
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/FIRRTL

  DEPENDS
  MLIRFIRRTLIncGen
  MLIRFIRRTLEnumsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTRTL
  MLIRIR
  MLIRPass
  )

add_dependencies(circt-headers MLIRFIRRTLIncGen MLIRFIRRTLEnumsIncGen)

add_subdirectory(Import)
add_subdirectory(Transforms)
