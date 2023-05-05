# circt-lec builds if a logic backend is found
if(CIRCT_LEC_ENABLED)
  add_llvm_tool(circt-lec
    circt-lec.cpp
  )

  target_link_libraries(circt-lec
    PRIVATE
    CIRCTSMT
    ${CIRCT_Z3_LIBS}
  )

  target_include_directories(circt-lec SYSTEM
    PRIVATE
    ${Z3_INCLUDE_DIR}
    ${Z3_CXX_INCLUDE_DIRS}
  )

  # Correct the runpath when linking shared libraries.
  if(BUILD_SHARED_LIBS)
    set_target_properties(circt-lec PROPERTIES
      INSTALL_RPATH_USE_LINK_PATH TRUE
    )
  endif()

  llvm_update_compile_flags(circt-lec)
  mlir_check_all_link_libraries(circt-lec)
endif()
