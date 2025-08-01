set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(circt_dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

if(MLIR_INCLUDE_TESTS)
  set(test_libs
    MLIRAffineTransformsTestPasses
    MLIRShapeTestPasses
    MLIRSPIRVTestPasses
    MLIRTestAnalysis
    MLIRTestDialect
    MLIRTestIR
    MLIRTestPass
    MLIRTestReducer
    MLIRTestRewrite
    MLIRTestTransforms
    )
endif()

set(LIBS
  ${circt_dialect_libs}
  ${conversion_libs}
  ${test_libs}
  MLIRAnalysis
  MLIRDialect
  MLIRLspServerLib
  MLIRParser
  MLIRPass
  MLIRRegisterAllDialects
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  )

add_circt_tool(circt-lsp-server
  circt-lsp-server.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(circt-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(circt-lsp-server)

mlir_check_all_link_libraries(circt-lsp-server)
