################################################################################
# Set up Python binding tools
################################################################################

include(AddMLIRPython)

################################################################################
# Declare native Python extension
################################################################################

declare_mlir_python_sources(CIRCTBindingsPythonExtension)

set(_depends LLVMSupport)

# TODO: remove these when ESIModule only uses C-APIs.
if (BUILD_SHARED_LIBS)
  list(APPEND _depends MLIRIR)
  list(APPEND _depends MLIRParser)
endif()

declare_mlir_python_extension(CIRCTBindingsPythonExtension.Core
  MODULE_NAME _circt
  ADD_TO_PARENT CIRCTBindingsPythonExtension
  SOURCES
    CIRCTModule.cpp
    ESIModule.cpp
    HWModule.cpp
    MSFTModule.cpp
  EMBED_CAPI_LINK_LIBS
    CIRCTCAPIComb
    CIRCTCAPIESI
    CIRCTCAPIMSFT
    CIRCTCAPIHW
    CIRCTCAPISeq
    CIRCTCAPISV
    CIRCTCAPIExportVerilog
  PRIVATE_LINK_LIBS
    ${_depends}
)

################################################################################
# Declare Python sources
################################################################################

declare_mlir_python_sources(CIRCTBindingsPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    circt/__init__.py
    circt/esi.py
    circt/support.py
    circt/dialects/_ods_common.py)

################################################################################
# Declare dialect-specific bindings.
################################################################################
declare_mlir_python_sources(CIRCTBindingsPythonSources.Dialects
  ADD_TO_PARENT CIRCTBindingsPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE circt/dialects/CombOps.td
  SOURCES
    circt/dialects/comb.py
    circt/dialects/_comb_ops_ext.py
  DIALECT_NAME comb)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE circt/dialects/ESIOps.td
  SOURCES circt/dialects/esi.py
  DIALECT_NAME esi)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE circt/dialects/HWOps.td
  SOURCES
    circt/dialects/hw.py
    circt/dialects/_hw_ops_ext.py
  DIALECT_NAME hw)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE circt/dialects/SeqOps.td
  SOURCES
    circt/dialects/seq.py
    circt/dialects/_seq_ops_ext.py
  DIALECT_NAME seq)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE circt/dialects/SVOps.td
  SOURCES circt/dialects/sv.py
  DIALECT_NAME sv)

################################################################################
# Build composite binaries
################################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(CIRCTBindingsPythonCAPI
  INSTALL_COMPONENT CIRCTBindingsPythonModules
  INSTALL_DESTINATION python_packages/circt_core/mlir/_mlir_libs
  # NOTE: When the MLIR API is relocated under circt, this would change to
  # .../circt/_mlir_libs
  OUTPUT_DIRECTORY "${CIRCT_PYTHON_PACKAGES_DIR}/circt_core/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    # TODO: This can be chopped down significantly for size.
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    CIRCTBindingsPythonSources
    CIRCTBindingsPythonExtension
)

# Bundle the MLIR python sources into our package.
# The MLIR API is position independent, so we explicitly output it to the mlir/
# folder as a temporary measure. It will eventually migrate under the circt/
# folder and be accessible under the unified "import circt..." namespace.
add_mlir_python_modules(CIRCTMLIRPythonModules
  ROOT_PREFIX "${CIRCT_PYTHON_PACKAGES_DIR}/circt_core/mlir"
  INSTALL_PREFIX "python_packages/circt_core/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    # We need the circt extensions co-located with the MLIR extensions. When
    # the namespace is unified, this moves to the below.
    CIRCTBindingsPythonExtension
  COMMON_CAPI_LINK_LIBS
    CIRCTBindingsPythonCAPI
)

# Bundle the CIRCT python sources into our package.
add_mlir_python_modules(CIRCTPythonModules
  ROOT_PREFIX "${CIRCT_PYTHON_PACKAGES_DIR}/circt_core"
  INSTALL_PREFIX "python_packages/circt_core"
  DECLARED_SOURCES
    CIRCTBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    CIRCTBindingsPythonCAPI
)
