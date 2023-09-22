##===- CMakeLists.txt - ESI dialect code ----------------------*- cmake -*-===//
##
## Implementation files the ESI dialect. Optionally include the Capnp
## dependent code.
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
  ESITranslations.cpp
  ESITypes.cpp
  Passes/ESIEmitCollateral.cpp
  Passes/ESILowerPhysical.cpp
  Passes/ESILowerPorts.cpp
  Passes/ESILowerToHW.cpp
  Passes/ESILowerTypes.cpp
  Passes/ESICleanMetadata.cpp

  APIUtilities.cpp

  # C++ API
  Passes/CPPAPI/ESIAddCPPAPI.cpp
  Passes/CPPAPI/ESIEmitCPPAPI.cpp
  Passes/CPPAPI/CPPAPI.cpp
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

if(CapnProto_FOUND)
  option(ESI_CAPNP "Enable ESI Capnp features" ON)
  list(APPEND srcs capnp/Schema.cpp)
  list(APPEND ESI_LinkLibs CapnProto::capnp CapnProto::capnpc)
endif()

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
  cosim/Cosim_DpiPkg.sv
  cosim/Cosim_Endpoint.sv
  cosim/Cosim_MMIO.sv
  ESIPrimitives.sv

  runtime/CMakeLists.txt
  runtime/cpp/include/esi/Accelerator.h
  runtime/cpp/include/esi/StdServices.h
  runtime/cpp/include/esi/backends/Cosim.h
  runtime/cpp/include/esi/backends/capnp.h
  runtime/cpp/include/esi/esi.h
  runtime/cpp/lib/Accelerator.cpp
  runtime/cpp/lib/StdServices.cpp
  runtime/cpp/lib/backends/Cosim.cpp
  runtime/cpp/tools/esiquery.cpp
  runtime/python/esi/esiaccel.cpp
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

if (ESI_CAPNP)
  add_dependencies(esi-collateral EsiCosimCapnp)
  get_target_property(EsiCosimCapnp_bindir EsiCosimCapnp SOURCE_DIR)
  set(ESI_COSIM_SCHEMA "${EsiCosimCapnp_bindir}/CosimDpi.capnp")
  set(CAPNPC_SRC_PREFIX "${EsiCosimCapnp_bindir}")
  add_custom_command(TARGET esi-collateral PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E
    copy "${ESI_COSIM_SCHEMA}"
         "${CMAKE_CURRENT_BINARY_DIR}/runtime/CosimDpi.capnp")

  target_compile_definitions(obj.CIRCTESI PRIVATE CAPNP)
  target_link_libraries(obj.CIRCTESI CapnProto::capnp CapnProto::capnpc)
endif()

add_subdirectory(cosim)

# We include this directory as collateral, but we also need to have a compiled
# version for testing.
option(ESI_RUNTIME "Build and test the ESI runtime")
llvm_canonicalize_cmake_booleans(ESI_RUNTIME)
if (ESI_RUNTIME)
  message(STATUS "Enabling ESI runtime build and testing")
  add_subdirectory(runtime)
endif()
