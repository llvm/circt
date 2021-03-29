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
    RTLModule.cpp
  LINK_LIBS
    CIRCTCAPIComb
    CIRCTCAPIRTL
    CIRCTCAPISV
)
add_dependencies(CIRCTBindingsPython CIRCTBindingsPythonExtension)

# TODO(mikeurbach): Temporary measures to be able to borrow upstream PyType.
target_include_directories(CIRCTBindingsPythonExtension PRIVATE
  ${MLIR_MAIN_SRC_DIR}/lib/Bindings/Python)
target_link_libraries(CIRCTBindingsPythonExtension PRIVATE
  MLIRCoreBindingsPythonExtension)

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

# The Python sources, including extensions, are in the circt.dialects namespace,
# as expected. But, at the moment, the bindings are generated in the
# mlir.dialects namespace, and look for Python extension modules there. To shim
# this, symlink the extension modules in mlir.dialects.
file(GLOB_RECURSE PY_EXT_FILES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/circt/dialects"
  "${CMAKE_CURRENT_SOURCE_DIR}/circt/*_ops_ext.py")
foreach(PY_EXT_FILE ${PY_EXT_FILES})
  set(PY_DEST_FILE "${PROJECT_BINARY_DIR}/python/mlir/dialects/${PY_EXT_FILE}")
  set(PY_SRC_FILE "${PROJECT_BINARY_DIR}/python/circt/dialects/${PY_EXT_FILE}")
  add_custom_command(
    TARGET CIRCTBindingsPythonSources POST_BUILD
    COMMENT "Symlinking Python extension ${PY_SRC_FILE} -> ${PY_DEST_FILE}"
    DEPENDS "${PY_SRC_FILE}"
    COMMAND "${CMAKE_COMMAND}" -E create_symlink "${PY_SRC_FILE}" "${PY_DEST_FILE}"
  )
endforeach()

################################################################################
# Generate dialect-specific bindings.
################################################################################
add_mlir_dialect_python_bindings(CIRCTBindingsPythonCombOps
  TD_FILE CombOps.td
  DIALECT_NAME comb)
add_dependencies(CIRCTBindingsPythonSources CIRCTBindingsPythonCombOps)

add_mlir_dialect_python_bindings(CIRCTBindingsPythonRTLOps
  TD_FILE RTLOps.td
  DIALECT_NAME rtl)
add_dependencies(CIRCTBindingsPythonSources CIRCTBindingsPythonRTLOps)

add_mlir_dialect_python_bindings(CIRCTBindingsPythonSVOps
  TD_FILE SVOps.td
  DIALECT_NAME sv)
add_dependencies(CIRCTBindingsPythonSources CIRCTBindingsPythonSVOps)
