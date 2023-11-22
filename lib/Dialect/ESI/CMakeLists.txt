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

set(ESI_RUNTIME_SRCS
  ESIPrimitives.sv

  runtime/cpp/include/esi/backends/Cosim.h
  runtime/cpp/include/esi/backends/Trace.h
  runtime/cpp/include/esi/Accelerator.h
  runtime/cpp/include/esi/Design.h
  runtime/cpp/include/esi/Manifest.h
  runtime/cpp/include/esi/StdServices.h
  runtime/cpp/include/esi/Types.h
  runtime/cpp/include/esi/Utils.h
  runtime/cpp/lib/backends/Cosim.cpp
  runtime/cpp/lib/backends/Trace.cpp
  runtime/cpp/lib/Accelerator.cpp
  runtime/cpp/lib/Design.cpp
  runtime/cpp/lib/Manifest.cpp
  runtime/cpp/lib/StdServices.cpp
  runtime/cpp/lib/Utils.cpp
  runtime/cpp/tools/esiquery.cpp
  runtime/cosim/CMakeLists.txt
  runtime/cosim/Cosim_MMIO.sv
  runtime/cosim/MtiPliStub/CMakeLists.txt
  runtime/cosim/MtiPliStub/DummySvDpi.cpp
  runtime/cosim/cosim_dpi_server/CMakeLists.txt
  runtime/cosim/cosim_dpi_server/Endpoint.cpp
  runtime/cosim/cosim_dpi_server/DpiEntryPoints.cpp
  runtime/cosim/cosim_dpi_server/Server.cpp
  runtime/cosim/include/cosim/LowLevel.h
  runtime/cosim/include/cosim/Utils.h
  runtime/cosim/include/cosim/Endpoint.h
  runtime/cosim/include/cosim/Server.h
  runtime/cosim/include/cosim/dpi.h
  runtime/cosim/include/dpi/svdpi.h
  runtime/cosim/Cosim_Endpoint.sv
  runtime/cosim/CosimDpi.capnp
  runtime/cosim/Cosim_DpiPkg.sv
  runtime/cosim/Cosim_Manifest.sv
  runtime/python/esi/__init__.py
  runtime/python/esi/accelerator.py
  runtime/python/esi/esiCppAccel.cpp
  runtime/python/esi/esiCppAccel.pyi
  runtime/CMakeLists.txt
)

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
