set(LLVM_LINK_COMPONENTS Support)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=circt.")

set(LLVM_OPTIONAL_SOURCES
  rtgtool.cpp
  Evaluator.cpp
)

add_circt_tool(rtgtool rtgtool.cpp)

add_library(rtgeval
  Evaluator.cpp
)

target_include_directories(rtgeval
  PRIVATE
  ${Python3_INCLUDE_DIRS}
)

target_link_libraries(rtgeval
  PRIVATE
  MLIRCAPIIR
  ${Python3_LIBRARIES}
)

target_link_libraries(rtgtool
  PRIVATE
  rtgeval
  CIRCTEmitRTGAssembly
  CIRCTRTGTestDialect
  CIRCTRTGDialect
  CIRCTRTGTransforms
  CIRCTSupport
  MLIRArithDialect
  MLIRIR
  MLIRPluginsLib
  LLVMSupport
)

llvm_update_compile_flags(rtgtool)
mlir_check_all_link_libraries(rtgtool)

configure_file(rtgentry.py ${CIRCT_TOOLS_DIR}/rtgentry.py)
add_custom_target(rtgentry SOURCES ${CIRCT_TOOLS_DIR}/rtgentry.py)
