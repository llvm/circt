set(VERSION_CPP "${CMAKE_CURRENT_BINARY_DIR}/Version.cpp")
set_source_files_properties("${VERSION_CPP}" PROPERTIES GENERATED TRUE)

add_circt_library(CIRCTSupport
  APInt.cpp
  BackedgeBuilder.cpp
  CustomDirectiveImpl.cpp
  Debug.cpp
  FVInt.cpp
  FieldRef.cpp
  InstanceGraph.cpp
  JSON.cpp
  LoweringOptions.cpp
  Naming.cpp
  ParsingUtils.cpp
  Passes.cpp
  Path.cpp
  PrettyPrinter.cpp
  PrettyPrinterHelpers.cpp
  SymCache.cpp
  TruthTable.cpp
  ValueMapper.cpp
  "${VERSION_CPP}"

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  )

#-------------------------------------------------------------------------------
# Generate Version.cpp
#-------------------------------------------------------------------------------

# Determine the path to a file that can be used as a dependency that changes
# whenever the git commit changes.
find_first_existing_vc_file("${CIRCT_SOURCE_DIR}" CIRCT_GIT_LOGS_HEAD)
set(GEN_VERSION_SCRIPT "${CIRCT_SOURCE_DIR}/cmake/modules/GenVersionFile.cmake")

# Generate the `Version.cpp` file. This either uses the latest git tag that
# matches a pattern if release tags are enabled, or the current git commit
# otherwise.
add_custom_command(
  OUTPUT "${VERSION_CPP}"
  DEPENDS "${CIRCT_GIT_LOGS_HEAD}" "${GEN_VERSION_SCRIPT}"
  COMMAND ${CMAKE_COMMAND}
  -DIN_FILE="${CMAKE_CURRENT_SOURCE_DIR}/Version.cpp.in"
  -DOUT_FILE="${VERSION_CPP}"
  -DRELEASE_TAG_PATTERN="${CIRCT_RELEASE_TAG}*"
  -DRELEASE_TAG_ENABLED=${CIRCT_RELEASE_TAG_ENABLED}
  -DSOURCE_ROOT="${CIRCT_SOURCE_DIR}"
  -P "${GEN_VERSION_SCRIPT}"
)
