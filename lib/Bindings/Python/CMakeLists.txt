################################################################################
# Set up Python binding tools
################################################################################

include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=circt.")

################################################################################
# Declare native Python extension
################################################################################


declare_mlir_python_extension(CIRCTBindingsPythonExtension
  MODULE_NAME _circt
  SOURCES
    CIRCTModule.cpp
    ESIModule.cpp
    HWModule.cpp
    OMModule.cpp
    MSFTModule.cpp
    SeqModule.cpp
    SVModule.cpp
  EMBED_CAPI_LINK_LIBS
    CIRCTCAPIComb
    CIRCTCAPIDebug
    CIRCTCAPIESI
    CIRCTCAPIExportVerilog
    CIRCTCAPIFSM
    CIRCTCAPIHW
    CIRCTCAPIHWArith
    CIRCTCAPIHandshake
    CIRCTCAPILTL
    CIRCTCAPIMSFT
    CIRCTCAPIOM
    CIRCTCAPISV
    CIRCTCAPISeq
    CIRCTCAPIVerif
    CIRCTCAPIConversion
    MLIRCAPITransforms
  PRIVATE_LINK_LIBS
    LLVMSupport
)

add_dependencies(CIRCTBindingsPythonExtension circt-headers)

################################################################################
# Declare Python sources
################################################################################

declare_mlir_python_sources(CIRCTBindingsPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
    support.py
)

################################################################################
# Declare dialect-specific bindings.
################################################################################

# Ensure the build directory for generated Python files exists. Ninja is able to
# generate this, but make does not and the build fails.
file(MAKE_DIRECTORY ${CIRCT_BINARY_DIR}/lib/Bindings/Python/circt/dialects)

declare_mlir_python_sources(CIRCTBindingsPythonSources.Dialects
  ADD_TO_PARENT CIRCTBindingsPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/CombOps.td
  SOURCES
    dialects/comb.py
  DIALECT_NAME comb)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/DebugOps.td
  SOURCES
    dialects/debug.py
  DIALECT_NAME dbg)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/ESIOps.td
  SOURCES
    dialects/esi.py
  DIALECT_NAME esi)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/HWOps.td
  SOURCES
    dialects/hw.py
  DIALECT_NAME hw)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/MSFTOps.td
  SOURCES
    dialects/msft.py
  DIALECT_NAME msft)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/OMOps.td
  SOURCES
    dialects/om.py
  DIALECT_NAME om)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/SeqOps.td
  SOURCES
    dialects/seq.py
  DIALECT_NAME seq)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/SVOps.td
  SOURCES
    dialects/sv.py
  DIALECT_NAME sv)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/FSMOps.td
  SOURCES
    dialects/fsm.py
  DIALECT_NAME fsm)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/HWArithOps.td
  SOURCES
    dialects/hwarith.py
  DIALECT_NAME hwarith)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/LTLOps.td
  SOURCES
    dialects/ltl.py
  DIALECT_NAME ltl)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/VerifOps.td
  SOURCES
    dialects/verif.py
  DIALECT_NAME verif)

################################################################################
# Build composite binaries
################################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(CIRCTBindingsPythonCAPI
  INSTALL_COMPONENT CIRCTPythonModules
  INSTALL_DESTINATION python_packages/circt_core/circt/_mlir_libs
  OUTPUT_DIRECTORY "${CIRCT_PYTHON_PACKAGES_DIR}/circt_core/circt/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources.Core
    CIRCTBindingsPythonExtension
)

# Bundle the CIRCT python sources into our package.
add_mlir_python_modules(CIRCTPythonModules
  ROOT_PREFIX "${CIRCT_PYTHON_PACKAGES_DIR}/circt_core/circt"
  INSTALL_PREFIX "python_packages/circt_core/circt"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    CIRCTBindingsPythonExtension
    CIRCTBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    CIRCTBindingsPythonCAPI
)
