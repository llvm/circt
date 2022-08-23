##===- CMakeLists.txt - Define a support library --------------*- cmake -*-===//
##
##===----------------------------------------------------------------------===//

set(VERSION_CPP "${CMAKE_CURRENT_BINARY_DIR}/Version.cpp")
set_source_files_properties("${VERSION_CPP}" PROPERTIES GENERATED TRUE)

set(LLVM_OPTIONAL_SOURCES
  BackedgeBuilder.cpp
  FieldRef.cpp
  LoweringOptions.cpp
  Path.cpp
  APInt.cpp
  ValueMapper.cpp
  SymCache.cpp
  TestPasses.cpp
  "${VERSION_CPP}"
  )


add_circt_library(CIRCTSupport
  BackedgeBuilder.cpp
  FieldRef.cpp
  LoweringOptions.cpp
  Path.cpp
  APInt.cpp
  ValueMapper.cpp
  SymCache.cpp
  "${VERSION_CPP}"

  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  MLIRIR
  )


add_circt_library(CIRCTSupportTestPasses
  TestPasses.cpp

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTESI
  CIRCTSeq
  MLIRPass
)


#-------------------------------------------------------------------------------
# Generate Version.cpp
#-------------------------------------------------------------------------------
find_first_existing_vc_file("${CIRCT_SOURCE_DIR}" CIRCT_GIT_LOGS_HEAD)
set(GEN_VERSION_SCRIPT "${CIRCT_SOURCE_DIR}/cmake/modules/GenVersionFile.cmake")

if (CIRCT_RELEASE_TAG_ENABLED)
  add_custom_command(OUTPUT "${VERSION_CPP}"
    DEPENDS "${CIRCT_GIT_LOGS_HEAD}" "${GEN_VERSION_SCRIPT}"
    COMMAND ${CMAKE_COMMAND} -DIN_FILE="${CMAKE_CURRENT_SOURCE_DIR}/Version.cpp.in"
    -DOUT_FILE="${VERSION_CPP}" -DRELEASE_PATTERN=${CIRCT_RELEASE_TAG}*
    -DDRY_RUN=OFF -DSOURCE_ROOT="${CIRCT_SOURCE_DIR}"
    -P "${GEN_VERSION_SCRIPT}")
else ()
  # If the release tag generation is disabled, run the script only at the first
  # cmake configuration.
  add_custom_command(OUTPUT "${VERSION_CPP}"
    DEPENDS "${GEN_VERSION_SCRIPT}"
    COMMAND ${CMAKE_COMMAND} -DIN_FILE="${CMAKE_CURRENT_SOURCE_DIR}/Version.cpp.in"
    -DOUT_FILE="${VERSION_CPP}" -DDRY_RUN=ON -DSOURCE_ROOT="${CIRCT_SOURCE_DIR}"
    -P "${GEN_VERSION_SCRIPT}")
endif()
