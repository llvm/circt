##===- CMakeLists.txt - Define a support library --------------*- cmake -*-===//
##
##===----------------------------------------------------------------------===//

set(VERSION_CPP "${CMAKE_CURRENT_BINARY_DIR}/Version.cpp")
set_source_files_properties("${VERSION_CPP}" PROPERTIES GENERATED TRUE)

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

#-------------------------------------------------------------------------------
# Generate Version.cpp
#-------------------------------------------------------------------------------
find_first_existing_vc_file("${CIRCT_SOURCE_DIR}" CIRCT_GIT_LOGS_HEAD)
set(GEN_VERSION_SCRIPT "${CIRCT_SOURCE_DIR}/cmake/modules/GenVersionFile.cmake")

# TODO: Change release tag pattern after public release
add_custom_command(OUTPUT "${VERSION_CPP}"
  DEPENDS "${CIRCT_GIT_LOGS_HEAD}" "${GEN_VERSION_SCRIPT}"
  COMMAND ${CMAKE_COMMAND} -DIN_FILE="${CMAKE_CURRENT_SOURCE_DIR}/Version.cpp.in"
  -DOUT_FILE="${VERSION_CPP}" -DRELEASE_PATTERN=circtorg* -P "${GEN_VERSION_SCRIPT}")
