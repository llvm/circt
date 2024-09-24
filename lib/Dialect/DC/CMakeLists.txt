set(DC_SRCS
  DCDialect.cpp
  DCOps.cpp
  DCTypes.cpp
  )

add_circt_dialect_library(CIRCTDC
  ${DC_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  DEPENDS
  MLIRArithDialect
  MLIRDCIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  CIRCTESI
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

add_dependencies(circt-headers
  MLIRDCIncGen
)

add_subdirectory(Transforms)
