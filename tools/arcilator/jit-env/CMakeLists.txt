add_library(arc-jit-env SHARED arcilator-jit-env.cpp)

if(WIN32)
# Put the DLL into the binary directory so we don't have
# to worry about it not being found.
set_target_properties(arc-jit-env
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        RUNTIME_OUTPUT_DIRECTORY ${CIRCT_TOOLS_DIR}$<0:>
        CXX_VISIBILITY_PRESET "default"
)
install(TARGETS arc-jit-env
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT arc-jit-env
)

else()

set_target_properties(arc-jit-env
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        CXX_VISIBILITY_PRESET "default"
)
install(TARGETS arc-jit-env
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT arc-jit-env
)

endif()

