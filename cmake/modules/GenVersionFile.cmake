# This script determines the project version and configures a `Version.cpp` file
# accordingly. (Based roughly on LLVM's `GenerateVersionFromVCS.cmake` and
# `VersionFromVCS.cmake`.)
#
# Run this using `cmake -P GenVersionFile.cmake <input-vars>`.
#
# Input variables:
#
# - `IN_FILE`:              Absolute path of `Version.cpp.in`
# - `OUT_FILE`:             Absolute path of `Version.cpp`
# - `RELEASE_TAG_PATTERN`:  A pattern like `firtool*` to search release tags
# - `RELEASE_TAG_ENABLED`:  Use release tag if true, or git hash if false
# - `SOURCE_ROOT`:          Path to root git directory

# Set the default version in case we fail to determine it.
set(CIRCT_VERSION "unknown git version")

# Determine the version using git.
find_package(Git QUIET)
if (Git_FOUND)
  # Either use `git describe` to find a matching tag, or `rev-parse` to get a
  # short hash of the current commit HEAD.
  if (RELEASE_TAG_ENABLED)
    set(git_cmd describe --dirty --tags --match "${RELEASE_TAG_PATTERN}")
  else()
    set(git_cmd rev-parse --short HEAD)
  endif()

  # Run the git command.
  string(JOIN " " git_cmd_str ${git_cmd})
  message(STATUS "Determining version with `git ${git_cmd_str}`")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} ${git_cmd}
    WORKING_DIRECTORY "${SOURCE_ROOT}"
    RESULT_VARIABLE git_exitcode
    OUTPUT_VARIABLE git_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # If the git command was successful, use its output as CIRCT version.
  # Otherwise display a warning and keep the version at its default.
  if (NOT ${git_exitcode})
    set(CIRCT_VERSION "${git_output}")
  else()
    message(WARNING "Cannot determine CIRCT version -- git command failed")
  endif ()
else()
  message(WARNING "Cannot determine CIRCT version -- git not found")
endif()

# Display the version we have found.
message(STATUS "CIRCT version: ${CIRCT_VERSION}")

# Replace the `@...@` placeholders in the input file. We use a temporary file to
# hold the output, such that we can only overwrite the output file if the
# version changed.
#
# (This command will prepend CMAKE_CURRENT_{SOURCE,BINARY}_DIR if <input> or
# <output> is relative, which is why IN_FILE and OUT_FILE must be absolute.)
set(tmpfile "${OUT_FILE}.tmp")
configure_file("${IN_FILE}" "${tmpfile}")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${tmpfile}" "${OUT_FILE}")
file(REMOVE "${OUT_FILE}.tmp")
