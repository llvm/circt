# SPT project.
set(CIRT_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include ) # --src-root
set(CIRT_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include ) # --includedir

set(CIRT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CIRT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")


## Should get this from MLIR build files somehow?
set(MLIR_TABLEGEN_EXE mlir-tblgen)
set(MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/mlir)
set(MLIR_MAIN_SRC_DIR ${MLIR_SOURCE_DIR})
set(MLIR_INCLUDE_DIR ${MLIR_SOURCE_DIR}/include )
set(MLIR_BINARY_INCLUDE_DIR ${LLVM_TOOLS_BINARY_DIR}/../tools/mlir/include )


# Installing the headers and docs needs to depend on generating any public
# tablegen'd targets.
add_custom_target(spt-headers)
set_target_properties(spt-headers PROPERTIES FOLDER "Misc")
add_custom_target(spt-doc)

include_directories( ${MLIR_INCLUDE_DIR})
include_directories( ${MLIR_BINARY_INCLUDE_DIR})
include_directories( ${CIRT_INCLUDE_DIR})
include_directories( ${CIRT_MAIN_SRC_DIR})

add_subdirectory(include/cirt)
add_subdirectory(lib)
add_subdirectory(tools)
#add_subdirectory(unittests)
add_subdirectory(test)

#option(CIRT_INCLUDE_DOCS "Generate build targets for the SPT docs."
#  ${LLVM_INCLUDE_DOCS} ${MLIR_INCLUDE_DOCS})
#if (CIRT_INCLUDE_DOCS)
#  add_subdirectory(docs)
#endif()

install(DIRECTORY include/spt
  DESTINATION include
  COMPONENT spt-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "LICENSE.TXT"
  )

install(DIRECTORY ${CIRT_INCLUDE_DIR}/spt
  DESTINATION include
  COMPONENT spt-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "config.h" EXCLUDE
  )

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-spt-headers
                           DEPENDS spt-headers
                           COMPONENT spt-headers)
endif()

add_subdirectory(cmake/modules)
