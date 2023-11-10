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

  APIUtilities.cpp
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

# Create a package of ESI collateral for any compiler to include with its
# distribution.
add_custom_target(esi-collateral)

file(GLOB_RECURSE
  ESI_RUNTIME_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*/*.*
)
list(APPEND ESI_RUNTIME_SRCS ESIPrimitives.sv)

# A function which copies a file from the source tree to the build tree.
# This is an alternative to configure_file to ensure that the file is copied
# on every build (not just during CMake configure).
function(copy_esi_runtime FILENAME)
  get_filename_component(SRC ${FILENAME} REALPATH)
  add_custom_command(TARGET esi-collateral PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E
    copy ${SRC} "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}")
  target_sources(esi-collateral PRIVATE "${FILENAME}")
endfunction()

foreach(SRC ${ESI_RUNTIME_SRCS})
  copy_esi_runtime(${SRC})
endforeach()

# We include this directory as collateral, but we also need to have a compiled
# version for testing.
option(ESI_RUNTIME "Build and test the ESI runtime" OFF)
llvm_canonicalize_cmake_booleans(ESI_RUNTIME)
if (ESI_RUNTIME)
  message(STATUS "Enabling ESI runtime build and testing")
  add_subdirectory(runtime)
endif()
