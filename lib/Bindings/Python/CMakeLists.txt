################################################################################
# Set up Python binding tools
################################################################################

include(AddMLIRPythonExtension)
add_custom_target(CIRCTBindingsPython)

################################################################################
# Build native Python extension
################################################################################

add_mlir_python_extension(CIRCTBindingsPythonExtension _circt
  INSTALL_DIR
    python
  SOURCES
    CIRCTModule.cpp
)
add_dependencies(CIRCTBindingsPython CIRCTBindingsPythonExtension)

################################################################################
# Python source tree management
################################################################################

set(PY_SRC_FILES
  circt/__init__.py
)

add_custom_target(CIRCTBindingsPythonSources ALL
  DEPENDS ${PY_SRC_FILES}
)
add_dependencies(CIRCTBindingsPython CIRCTBindingsPythonSources)

foreach(PY_SRC_FILE ${PY_SRC_FILES})
  set(PY_DEST_FILE "${PROJECT_BINARY_DIR}/python/${PY_SRC_FILE}")
  add_custom_command(
    TARGET CIRCTBindingsPythonSources PRE_BUILD
    COMMENT "Copying Python source ${PY_SRC_FILE} -> ${PY_DEST_FILE}"
    DEPENDS "${PY_SRC_FILE}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/${PY_SRC_FILE}" "${PY_DEST_FILE}"
  )
endforeach()
