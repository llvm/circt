################################################################################
# Set up Python binding tools
################################################################################

include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=circt.")

################################################################################
# Declare native Python extension
################################################################################
set(LLVM_OPTIONAL_SOURCES
  RTGTestModule.cpp
)

set(PYTHON_BINDINGS_SOURCES
  AIGModule.cpp
  CIRCTModule.cpp
  ESIModule.cpp
  HWModule.cpp
  MSFTModule.cpp
  OMModule.cpp
  RTGModule.cpp
  SeqModule.cpp
  SupportModule.cpp
  SVModule.cpp
  # Headers must be included explicitly so they are installed.
  CIRCTModules.h
  NanobindUtils.h
)

set(PYTHON_BINDINGS_LINK_LIBS
  CIRCTCAPIAIG
  CIRCTCAPIArc
  CIRCTCAPIComb
  CIRCTCAPIConversion
  CIRCTCAPIDC
  CIRCTCAPIDebug
  CIRCTCAPIEmit
  CIRCTCAPIESI
  CIRCTCAPIExportVerilog
  CIRCTCAPIFSM
  CIRCTCAPIHandshake
  CIRCTCAPIHW
  CIRCTCAPIHWArith
  CIRCTCAPIKanagawa
  CIRCTCAPILTL
  CIRCTCAPIMSFT
  CIRCTCAPIOM
  CIRCTCAPIPipeline
  CIRCTCAPIRTG
  CIRCTCAPISeq
  CIRCTCAPISV
  CIRCTCAPISupport
  CIRCTCAPIVerif
  CIRCTCAPITransforms
  CIRCTCAPISynthesis
  MLIRCAPIIndex
  MLIRCAPISMT
  MLIRCAPIExportSMTLIB
  MLIRCAPISCF
  # needed for mlirFrozenRewritePatternSetDestroy
  # but not the actual passes
  MLIRCAPITransforms
)

if (CIRCT_INCLUDE_TESTS)
  list(APPEND PYTHON_BINDINGS_SOURCES RTGTestModule.cpp)
  list(APPEND PYTHON_BINDINGS_LINK_LIBS CIRCTCAPIRTGTest)
endif()

declare_mlir_python_extension(CIRCTBindingsPythonExtension
  MODULE_NAME _circt
  SOURCES
    ${PYTHON_BINDINGS_SOURCES}
  EMBED_CAPI_LINK_LIBS
    ${PYTHON_BINDINGS_LINK_LIBS}
  PRIVATE_LINK_LIBS
    LLVMSupport
  PYTHON_BINDINGS_LIBRARY
    nanobind
)

if (CIRCT_INCLUDE_TESTS)
  target_compile_definitions(CIRCTBindingsPythonExtension INTERFACE CIRCT_INCLUDE_TESTS)
endif()

add_dependencies(CIRCTBindingsPythonExtension circt-headers)

################################################################################
# Declare Python sources
################################################################################

declare_mlir_python_sources(CIRCTBindingsPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
)

declare_mlir_python_sources(CIRCTBindingsPythonSources.Support
  ADD_TO_PARENT CIRCTBindingsPythonSources
  SOURCES
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
  TD_FILE dialects/AIGOps.td
  SOURCES
    dialects/aig.py
  DIALECT_NAME aig)

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
  TD_FILE dialects/HandshakeOps.td
  SOURCES
    dialects/handshake.py
  DIALECT_NAME handshake)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/HWOps.td
  SOURCES
    dialects/hw.py
  DIALECT_NAME hw)

# We need the 'arith.py' file because 'scf.py' imports from it. We are not
# calling the function that constructs arith ops and thus don't need to
# register and link against the dialect.
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${MLIR_MAIN_SRC_DIR}/python/mlir"
  TD_FILE dialects/ArithOps.td
  SOURCES
    dialects/arith.py
  DIALECT_NAME arith
  GEN_ENUM_BINDINGS)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${MLIR_MAIN_SRC_DIR}/python/mlir"
  TD_FILE dialects/IndexOps.td
  SOURCES
    dialects/index.py
  DIALECT_NAME index
  GEN_ENUM_BINDINGS)

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
  TD_FILE dialects/RTGOps.td
  SOURCES
    dialects/rtg.py
  DIALECT_NAME rtg)

if (CIRCT_INCLUDE_TESTS)
  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    TD_FILE dialects/RTGTestOps.td
    SOURCES
      dialects/rtgtest.py
    DIALECT_NAME rtgtest)
endif()

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

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${MLIR_MAIN_SRC_DIR}/python/mlir"
  TD_FILE dialects/SCFOps.td
  SOURCES
    dialects/scf.py
  DIALECT_NAME scf)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/EmitOps.td
  SOURCES
    dialects/emit.py
  DIALECT_NAME emit)

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
    MLIRPythonSources.Dialects.smt
    CIRCTBindingsPythonExtension
    CIRCTBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    CIRCTBindingsPythonCAPI
)
