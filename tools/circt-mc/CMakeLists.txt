# circt-mc builds if a logic backend is found
if(CIRCT_LEC_ENABLED)
  add_llvm_tool(circt-mc
    circt-mc.cpp
  )

  target_link_libraries(circt-mc
    PRIVATE
    CIRCTLogicalEquivalence
    ${Z3_LIBRARIES}
  )

  # Correct the runpath when linking shared libraries.
  if(BUILD_SHARED_LIBS)
    set_target_properties(circt-mc PROPERTIES
      INSTALL_RPATH_USE_LINK_PATH TRUE
    )
  endif()

  llvm_update_compile_flags(circt-mc)
  mlir_check_all_link_libraries(circt-mc)
endif()
