################################################################################
# Set up Python binding tools
################################################################################

include(AddMLIRPython)
add_custom_target(CIRCTBindingsPython)

################################################################################
# Build native Python extension
################################################################################

add_mlir_python_extension(CIRCTBindingsPythonExtension _circt
  INSTALL_DIR
    python
  SOURCES
    CIRCTModule.cpp
    ESIModule.cpp
    HWModule.cpp
    MSFTModule.cpp
  LINK_LIBS
    CIRCTCAPIComb
    CIRCTCAPIESI
    CIRCTCAPIMSFT
    CIRCTCAPIHW
    CIRCTCAPISeq
    CIRCTCAPISV
    CIRCTCAPIExportVerilog
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

# Note that we copy from the source tree just like for headers because
# it will not be polluted with py_cache runtime artifacts (from testing and
# such).
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/circt
  DESTINATION python
  COMPONENT CIRCTBindingsPythonSources
  FILES_MATCHING PATTERN "*.py"
)

# Dialect sources are generated. Install separately.
# Note that __pycache__ directories may have been left by tests and other
# executions. And __init__.py is handled as a regular source file.
install(
  DIRECTORY ${PROJECT_BINARY_DIR}/python/mlir/dialects
  DESTINATION python/mlir
  COMPONENT CIRCTBindingsPythonDialects
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
  PATTERN "__init__.py" EXCLUDE
)

################################################################################
# Generate dialect-specific bindings.
################################################################################
add_mlir_dialect_python_bindings(CIRCTBindingsPythonCombOps
  TD_FILE CombOps.td
  DIALECT_NAME comb)
add_dependencies(CIRCTBindingsPythonSources CIRCTBindingsPythonCombOps)

add_mlir_dialect_python_bindings(CIRCTBindingsPythonESIOps
  TD_FILE ESIOps.td
  DIALECT_NAME esi)
add_dependencies(CIRCTBindingsPythonSources CIRCTBindingsPythonESIOps)

add_mlir_dialect_python_bindings(CIRCTBindingsPythonHWOps
  TD_FILE HWOps.td
  DIALECT_NAME rtl)
add_dependencies(CIRCTBindingsPythonSources CIRCTBindingsPythonHWOps)

add_mlir_dialect_python_bindings(CIRCTBindingsPythonSeqOps
  TD_FILE SeqOps.td
  DIALECT_NAME seq)
add_dependencies(CIRCTBindingsPythonSources CIRCTBindingsPythonSeqOps)

add_mlir_dialect_python_bindings(CIRCTBindingsPythonSVOps
  TD_FILE SVOps.td
  DIALECT_NAME sv)
add_dependencies(CIRCTBindingsPythonSources CIRCTBindingsPythonSVOps)
