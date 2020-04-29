# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(CIRT_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(cirt_cmake_builddir "${CMAKE_BINARY_DIR}/${CIRT_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

get_property(CIRT_EXPORTS GLOBAL PROPERTY CIRT_EXPORTS)
export(TARGETS ${CIRT_EXPORTS} FILE ${cirt_cmake_builddir}/CIRTTargets.cmake)

# Generate CIRTConfig.cmake for the build tree.
set(CIRT_CONFIG_CMAKE_DIR "${cirt_cmake_builddir}")
set(CIRT_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(CIRT_CONFIG_EXPORTS_FILE "${cirt_cmake_builddir}/CIRTTargets.cmake")
set(CIRT_CONFIG_INCLUDE_DIRS
  "${CIRT_SOURCE_DIR}/include"
  "${CIRT_BINARY_DIR}/include"
  )
set(CIRT_CONFIG_CMAKE_DIR)
set(CIRT_CONFIG_LLVM_CMAKE_DIR)
set(CIRT_CONFIG_EXPORTS_FILE)

# Generate CIRTConfig.cmake for the install tree.
set(CIRT_CONFIG_CODE "
# Compute the installation prefix from this LLVMConfig.cmake file location.
get_filename_component(CIRT_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${CIRT_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(CIRT_CONFIG_CODE "${CIRT_CONFIG_CODE}
get_filename_component(CIRT_INSTALL_PREFIX \"\${CIRT_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(CIRT_CONFIG_CMAKE_DIR "\${CIRT_INSTALL_PREFIX}/${CIRT_INSTALL_PACKAGE_DIR}")
set(CIRT_CONFIG_LLVM_CMAKE_DIR "\${CIRT_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
set(CIRT_CONFIG_EXPORTS_FILE "\${CIRT_CMAKE_DIR}/CIRTTargets.cmake")
set(CIRT_CONFIG_INCLUDE_DIRS
  "\${CIRT_INSTALL_PREFIX}/include"
  )
set(CIRT_CONFIG_CODE)
set(CIRT_CONFIG_CMAKE_DIR)
set(CIRT_CONFIG_EXPORTS_FILE)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  # Not TOOLCHAIN ONLY, so install the CIRT parts as well
  # Include the cmake files so other tools can use cirt-tblgen, etc.
  get_property(cirt_has_exports GLOBAL PROPERTY CIRT_HAS_EXPORTS)
  if(cirt_has_exports)
    install(EXPORT CIRTTargets DESTINATION ${CIRT_INSTALL_PACKAGE_DIR}
            COMPONENT cirt-cmake-exports)
  endif()

  install(FILES #${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CIRTConfig.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AddCIRT.cmake
    DESTINATION ${CIRT_INSTALL_PACKAGE_DIR}
    COMPONENT cirt-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(cirt-cmake-exports)
    add_llvm_install_targets(install-cirt-cmake-exports
                             COMPONENT cirt-cmake-exports)
  endif()
endif()
