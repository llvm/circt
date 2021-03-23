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
  LINK_LIBS
    CIRCTCAPIRTL
)
add_dependencies(CIRCTBindingsPython CIRCTBindingsPythonExtension)

################################################################################
# Python source tree management
################################################################################

file(GLOB_RECURSE PY_SRC_FILES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/circt/*.py")

add_custom_target(CIRCTBindingsPythonSources ALL
  DEPENDS ${PY_SRC_FILES}
)
add_dependencies(CIRCTBindingsPython CIRCTBindingsPythonSources)

foreach(PY_SRC_FILE ${PY_SRC_FILES})
  set(PY_DEST_FILE "${PROJECT_BINARY_DIR}/python/${PY_SRC_FILE}")
  get_filename_component(PY_DEST_DIR "${PY_DEST_FILE}" DIRECTORY)
  file(MAKE_DIRECTORY "${PY_DEST_DIR}")
  add_custom_command(
    TARGET CIRCTBindingsPythonSources PRE_BUILD
    COMMENT "Symlinking Python source ${PY_SRC_FILE} -> ${PY_DEST_FILE}"
    DEPENDS "${PY_SRC_FILE}"
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
        "${CMAKE_CURRENT_SOURCE_DIR}/${PY_SRC_FILE}" "${PY_DEST_FILE}"
  )
endforeach()

################################################################################
# Generate dialect-specific bindings.
################################################################################
add_mlir_dialect_python_bindings(CIRCTBindingsPythonRTLOps
  TD_FILE RTLOps.td
  DIALECT_NAME rtl)
add_dependencies(CIRCTBindingsPythonSources CIRCTBindingsPythonRTLOps)
