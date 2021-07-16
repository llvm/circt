# ===- CMakeLists.txt - PyCDE sources --------------------------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//

file(GLOB_RECURSE PY_SRC_FILES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/pycde/*.py")

add_custom_target(PyCDE ALL
  DEPENDS ${PY_SRC_FILES}
)

foreach(PY_SRC_FILE ${PY_SRC_FILES})
  set(PY_DEST_FILE "${PROJECT_BINARY_DIR}/python/${PY_SRC_FILE}")
  get_filename_component(PY_DEST_DIR "${PY_DEST_FILE}" DIRECTORY)
  file(MAKE_DIRECTORY "${PY_DEST_DIR}")
  add_custom_command(
    TARGET PyCDE PRE_BUILD
    COMMENT "Symlinking Python source ${PY_SRC_FILE} -> ${PY_DEST_FILE}"
    DEPENDS "${PY_SRC_FILE}"
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
        "${CMAKE_CURRENT_SOURCE_DIR}/${PY_SRC_FILE}" "${PY_DEST_FILE}"
  )
endforeach()

# Note that we copy from the source tree just like for headers because
# it will not be polluted with py_cache runtime artifacts (from testing and
# such).
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pycde
  DESTINATION python
  COMPONENT PyCDE
  FILES_MATCHING PATTERN "*.py"
)
