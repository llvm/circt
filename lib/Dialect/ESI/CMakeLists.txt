##===- CMakeLists.txt - ESI dialect code ----------------------*- cmake -*-===//
##
## Implementation files the ESI dialect. Optionally include the Capnp
## dependent code.
##
##===----------------------------------------------------------------------===//

set(srcs
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

  cosim/APIUtilities.cpp

  # C++ cosimulation API
  cosim/CPP/ESIAddCPPCosimAPI.cpp
  cosim/CPP/ESIEmitCPPCosimAPI.cpp
  cosim/CPP/CPPCosimAPI.cpp
)

set(ESI_RUNTIME_DIR ${CIRCT_BINARY_DIR}/include/circt/Dialect/ESI)

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

# configure_file all files in the 'runtime' subdirectory, recursively.
# The output file should reside in ${ESI_RUNTIME_DIR} and have the same
# relative path as the input file.
function(esi_configure_runtime_files)
  foreach(file ${ARGN})
    get_filename_component(dir ${file} DIRECTORY)
    get_filename_component(name ${file} NAME)
    string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR} "" dir ${dir})
    set(output ${ESI_RUNTIME_DIR}/${dir}/${name})
    configure_file(${file} ${output} COPYONLY)
  endforeach()
endfunction()

# Configure all files in the 'runtime' subdirectory.
file(GLOB_RECURSE ESI_RUNTIME_FILES ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*)
esi_configure_runtime_files(${ESI_RUNTIME_FILES})


set(CP_FILES
  ESIPrimitives.sv
)

if(CapnProto_FOUND)
  option(ESI_CAPNP "Enable ESI Capnp features" ON)
endif()

if (ESI_CAPNP)
  list(APPEND srcs
    capnp/Schema.cpp
  )
  list(APPEND ESI_LinkLibs CapnProto::capnp CapnProto::capnpc)
  list(APPEND CP_FILES ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/ESI/cosim/CosimDpi.capnp)
  list(APPEND CP_FILES cosim/Cosim_DpiPkg.sv)
  list(APPEND CP_FILES cosim/Cosim_Endpoint.sv)
  install(FILES ${CapnProto_capnp_IMPORTED_LOCATION} DESTINATION lib)
  install(FILES ${CapnProto_capnpc_IMPORTED_LOCATION} DESTINATION lib)
  install(FILES ${CapnProto_kj_IMPORTED_LOCATION} DESTINATION lib)
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

foreach(CP_FILE ${CP_FILES})
  get_filename_component(DST ${CP_FILE} NAME)
  get_filename_component(SRC ${CP_FILE} REALPATH)
  list(APPEND ABS_CFILES ${SRC})
  add_custom_command(TARGET CIRCTESI PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E
    copy ${SRC} ${CIRCT_BINARY_DIR}/${DST})
endforeach()

# Create a package of ESI collateral for any compiler to include with its
# distribution.
add_library(esi-collateral INTERFACE)
set_target_properties(esi-collateral PROPERTIES PUBLIC_HEADER "${ABS_CFILES}")

if (ESI_CAPNP)
  target_compile_definitions(obj.CIRCTESI PRIVATE CAPNP)
  target_link_libraries(obj.CIRCTESI CapnProto::capnp CapnProto::capnpc)
endif()

add_subdirectory(cosim)
