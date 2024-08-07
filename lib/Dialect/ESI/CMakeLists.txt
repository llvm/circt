##===- CMakeLists.txt - ESI dialect code ----------------------*- cmake -*-===//
##
## Implementation files the ESI dialect.
##
##===----------------------------------------------------------------------===//

set(srcs
  AppID.cpp
  ESIAttributes.cpp
  ESIDialect.cpp
  ESIFolds.cpp
  ESIOps.cpp
  ESIPasses.cpp
  ESIServices.cpp
  ESIStdServices.cpp
  ESITypes.cpp
  Passes/ESILowerPhysical.cpp
  Passes/ESILowerBundles.cpp
  Passes/ESILowerPorts.cpp
  Passes/ESILowerToHW.cpp
  Passes/ESILowerTypes.cpp
  Passes/ESICleanMetadata.cpp
  Passes/ESIBuildManifest.cpp
  Passes/ESIAppIDHier.cpp
  Passes/ESIVerifyConnections.cpp
)

set(ESI_LinkLibs
  CIRCTSupport
  CIRCTComb
  CIRCTMSFT
  CIRCTMSFTTransforms
  CIRCTSV
  CIRCTHW
  MLIRIR
  MLIRTransforms
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTranslateLib
)

set(ESI_Deps
  ${ESI_LinkLibs}
  MLIRESITransformsIncGen
  MLIRESIInterfacesIncGen
)

add_circt_dialect_library(CIRCTESI
  ${srcs}

  DEPENDS
  MLIRESITransformsIncGen
  MLIRESIEnumsIncGen
  ${ESI_Deps}

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC
  ${ESI_LinkLibs}
  )

add_custom_target(ESIPrimitives
SOURCES
  ESIPrimitives.sv
)
add_custom_command(TARGET ESIPrimitives POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/ESIPrimitives.sv
            ${CMAKE_CURRENT_BINARY_DIR}/ESIPrimitives.sv
)

option(ESI_RUNTIME "Build and test the ESI runtime" OFF)
llvm_canonicalize_cmake_booleans(ESI_RUNTIME)
if (ESI_RUNTIME)
  message(STATUS "Enabling ESI runtime build and testing")
  add_subdirectory(runtime)
endif()
