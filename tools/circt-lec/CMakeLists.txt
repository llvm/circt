# circt-lec builds if a logic backend is found
if(CIRCT_LEC_ENABLED)
  list(APPEND CIRCT_LEC_SOURCES
    circt-lec.cpp
    Circuit.cpp
    LogicExporter.cpp
    Solver.cpp
  )

  list(APPEND CIRCT_LEC_LIBS
    MLIRTransforms
    MLIRTranslateLib
    CIRCTComb
    CIRCTHW
    CIRCTSupport
  )

  add_llvm_tool(circt-lec ${CIRCT_LEC_SOURCES})
  target_link_libraries(circt-lec PRIVATE ${CIRCT_LEC_LIBS})
  target_include_directories(circt-lec PRIVATE ${CIRCT_LEC_INCLUDES})

  # Correct the runpath when linking shared libraries.
  if(BUILD_SHARED_LIBS)
    set_target_properties(circt-lec PROPERTIES
      INSTALL_RPATH_USE_LINK_PATH TRUE
    )
  endif()

  llvm_update_compile_flags(circt-lec)
  mlir_check_all_link_libraries(circt-lec)
endif()
