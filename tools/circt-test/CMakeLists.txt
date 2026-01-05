get_property(dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)

set(libs
  ${dialect_libs}

  CIRCTExportVerilog
  CIRCTSeqToSV
  CIRCTSimToSV
  CIRCTSVTransforms
  CIRCTVerifToSV
  CIRCTVerifTransforms

  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRSMT

  MLIRBytecodeReader
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_circt_tool(circt-test circt-test.cpp DEPENDS ${libs})
target_link_libraries(circt-test PRIVATE ${libs})

llvm_update_compile_flags(circt-test)
mlir_check_all_link_libraries(circt-test)

#-------------------------------------------------------------------------------
# Runners
#-------------------------------------------------------------------------------

add_custom_target(circt-test-runners)
add_custom_target(install-circt-test-runners)

foreach(runner sby circt-bmc)
  set(name circt-test-runner-${runner})

  # Copy the test runner script to the tools build directory.
  set(src_path ${CMAKE_CURRENT_SOURCE_DIR}/circt-test-runner-${runner}.py)
  set(dst_path ${CIRCT_TOOLS_DIR}/circt-test-runner-${runner}.py)
  add_custom_command(
    OUTPUT ${dst_path}
    DEPENDS ${src_path}
    COMMAND ${CMAKE_COMMAND} -E copy ${src_path} ${dst_path}
  )
  add_custom_target(${name} DEPENDS ${dst_path})
  add_dependencies(circt-test-runners ${name})

  # Add an install target for the runner.
  if (CIRCT_BUILD_TOOLS)
    install(
      PROGRAMS ${dst_path}
      DESTINATION "${CIRCT_TOOLS_INSTALL_DIR}"
      COMPONENT ${name}
    )
    if(NOT CMAKE_CONFIGURATION_TYPES)
      add_llvm_install_targets(install-${name}
        DEPENDS ${name}
        COMPONENT ${name})
    endif()
    add_dependencies(install-circt-test-runners install-${name})
  endif()
endforeach()

add_dependencies(circt-test circt-test-runners)
add_dependencies(install-circt-test install-circt-test-runners)
