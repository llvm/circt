# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(SPT_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(spt_cmake_builddir "${CMAKE_BINARY_DIR}/${SPT_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

get_property(SPT_EXPORTS GLOBAL PROPERTY SPT_EXPORTS)
export(TARGETS ${SPT_EXPORTS} FILE ${spt_cmake_builddir}/SptTargets.cmake)

# Generate SptConfig.cmake for the build tree.
set(SPT_CONFIG_CMAKE_DIR "${spt_cmake_builddir}")
set(SPT_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(SPT_CONFIG_EXPORTS_FILE "${spt_cmake_builddir}/SptTargets.cmake")
set(SPT_CONFIG_INCLUDE_DIRS
  "${SPT_SOURCE_DIR}/include"
  "${SPT_BINARY_DIR}/include"
  )
set(SPT_CONFIG_CMAKE_DIR)
set(SPT_CONFIG_LLVM_CMAKE_DIR)
set(SPT_CONFIG_EXPORTS_FILE)

# Generate SptConfig.cmake for the install tree.
set(SPT_CONFIG_CODE "
# Compute the installation prefix from this LLVMConfig.cmake file location.
get_filename_component(SPT_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${SPT_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(SPT_CONFIG_CODE "${SPT_CONFIG_CODE}
get_filename_component(SPT_INSTALL_PREFIX \"\${SPT_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(SPT_CONFIG_CMAKE_DIR "\${SPT_INSTALL_PREFIX}/${SPT_INSTALL_PACKAGE_DIR}")
set(SPT_CONFIG_LLVM_CMAKE_DIR "\${SPT_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
set(SPT_CONFIG_EXPORTS_FILE "\${SPT_CMAKE_DIR}/SptTargets.cmake")
set(SPT_CONFIG_INCLUDE_DIRS
  "\${SPT_INSTALL_PREFIX}/include"
  )
set(SPT_CONFIG_CODE)
set(SPT_CONFIG_CMAKE_DIR)
set(SPT_CONFIG_EXPORTS_FILE)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  # Not TOOLCHAIN ONLY, so install the SPT parts as well
  # Include the cmake files so other tools can use spt-tblgen, etc.
  get_property(spt_has_exports GLOBAL PROPERTY SPT_HAS_EXPORTS)
  if(spt_has_exports)
    install(EXPORT SptTargets DESTINATION ${SPT_INSTALL_PACKAGE_DIR}
            COMPONENT spt-cmake-exports)
  endif()

  install(FILES #${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/SptConfig.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AddSPT.cmake
    DESTINATION ${SPT_INSTALL_PACKAGE_DIR}
    COMPONENT spt-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(spt-cmake-exports)
    add_llvm_install_targets(install-spt-cmake-exports
                             COMPONENT spt-cmake-exports)
  endif()
endif()
