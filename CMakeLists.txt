##===- CMakeLists.txt - CIRCT cmake root ----------------------*- cmake -*-===//
##
## Configure the CIRCT build.
##
##===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.20.0)

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

  project(circt LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# If we are not building as a part of LLVM, build Circt as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  if (CIRCT_BINDINGS_PYTHON_ENABLED)
    message(FATAL_ERROR "CIRCT Python bindings require a unified build. \
                         See docs/PythonBindings.md.")
  endif()

# Generate a CompilationDatabase (compile_commands.json file) for our build,
# for use by clang_complete, YouCompleteMe, etc.
  set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

  if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c- /GR-")
  else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")
  endif ()

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------

  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  set(CIRCT_BUILT_STANDALONE 1)
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")

  # Handle unittests when building out-of-tree against an installed version of
  # LLVM/MLIR (not a build tree). Adapted from `llvm/flang/CMakeLists.txt`.
  set(CIRCT_GTEST_AVAILABLE 0)
  if (TARGET llvm_gtest)
    # Installed gtest, via LLVM_INSTALL_GTEST.  Preferred.
    message(STATUS "LLVM GTest found, enabling unittests")
    set(CIRCT_GTEST_AVAILABLE 1)
  else()
    set(UNITTEST_DIR ${LLVM_THIRD_PARTY_DIR}/unittest)
    if (NOT EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
      set(UNITTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm/third-party/unittest)
    endif()
    if (EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
      find_package(Threads)
      add_llvm_library(llvm_gtest
        ${UNITTEST_DIR}/googletest/src/gtest-all.cc
        ${UNITTEST_DIR}/googlemock/src/gmock-all.cc
        LINK_COMPONENTS Support # llvm::raw_ostream
        BUILDTREE_ONLY
      )
      target_include_directories(llvm_gtest
        PUBLIC
        "${UNITTEST_DIR}/googletest/include"
        "${UNITTEST_DIR}/googlemock/include"
        PRIVATE
        "${UNITTEST_DIR}/googletest"
        "${UNITTEST_DIR}/googlemock"
      )
      target_link_libraries(llvm_gtest PUBLIC Threads::Threads)
      add_llvm_library(llvm_gtest_main
        ${UNITTEST_DIR}/UnitTestMain/TestMain.cpp
        LINK_LIBS llvm_gtest
        LINK_COMPONENTS Support # llvm::cl
        BUILDTREE_ONLY
      )
      set(CIRCT_GTEST_AVAILABLE 1)
    else()
      message(WARNING "Skipping unittests since LLVM install does not include \
        gtest headers and libraries")
      set(CIRCT_GTEST_AVAILABLE 0)
    endif()
  endif()

else()
  # CMake library generation settings.
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Default to building a static mondo-lib")
  set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON CACHE BOOL
    "Python soname linked libraries are bad")
  set(CMAKE_VISIBILITY_INLINES_HIDDEN ON CACHE BOOL "Hide inlines")

  # The -fvisibility=hidden option only works for static builds.
  if (BUILD_SHARED_LIBS AND (CMAKE_CXX_VISIBILITY_PRESET STREQUAL "hidden"))
    message(FATAL_ERROR "CMAKE_CXX_VISIBILITY_PRESET=hidden is incompatible \
                         with BUILD_SHARED_LIBS.")
  endif()

  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir ) # --src-root
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include ) # --includedir
  set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

  # If building as part of a unified build, whether or not MLIR's execution engine
  # is enabled must be fetched from its subdirectory scope.
  get_directory_property(MLIR_ENABLE_EXECUTION_ENGINE
      DIRECTORY ${MLIR_MAIN_SRC_DIR}
      DEFINITION MLIR_ENABLE_EXECUTION_ENGINE)

  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")

  set(CIRCT_GTEST_AVAILABLE 1)
endif()

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")
if (MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

#-------------------------------------------------------------------------------
# CIRCT configuration
#-------------------------------------------------------------------------------

# CIRCT project.
set(CIRCT_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} ) # --src-root
set(CIRCT_MAIN_INCLUDE_DIR ${CIRCT_MAIN_SRC_DIR}/include)

set(CIRCT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CIRCT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CIRCT_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(CIRCT_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)
set(CIRCT_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(CIRCT_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)
set(CIRCT_PYTHON_PACKAGES_DIR ${CIRCT_BINARY_DIR}/python_packages)


option(CIRCT_INCLUDE_TOOLS "Generate build targets for the CIRCT tools." ON)
option(CIRCT_BUILD_TOOLS "Build the CIRCT tools. If OFF, just generate build targets." ON)
set(CIRCT_TOOLS_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
    "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')")

option(CIRCT_INCLUDE_TESTS
       "Generate build targets for the CIRCT tests." ON)

list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddCIRCT)

# Installing the headers and docs needs to depend on generating any public
# tablegen'd targets.
add_custom_target(circt-headers)
set_target_properties(circt-headers PROPERTIES FOLDER "Misc")
add_custom_target(circt-doc)

# Umbrella target that builds all C API bindings.
add_custom_target(circt-capi)

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add CIRCT files to the include path
include_directories(${CIRCT_MAIN_INCLUDE_DIR})
include_directories(${CIRCT_INCLUDE_DIR})

# Set the release tag.
option(CIRCT_RELEASE_TAG_ENABLED "Emit the release tag to output." OFF)
if (NOT CIRCT_RELEASE_TAG_ENABLED)
  message(STATUS "Version generation is disabled. To enable the version "
               "generation, please set CIRCT_RELEASE_TAG_ENABLED CMake "
               "variable")
endif()

set(CIRCT_RELEASE_TAG "circtorg" CACHE STRING
    "Prefix of the release tag (e.g. circtorg, firtool, and pycde).")

#-------------------------------------------------------------------------------
# Verilator Configuration
#-------------------------------------------------------------------------------

# If Verilator hasn't been explicitly disabled, find it.
option(VERILATOR_DISABLE "Disable the Verilator tests.")
if (VERILATOR_DISABLE)
  message(STATUS "Disabling Verilator tests.")
else()
  # Detect if Verilator is present.
  if (NOT DEFINED VERILATOR_PATH)
    find_program(VERILATOR_PATH "verilator" PATHS
      "${CMAKE_CURRENT_SOURCE_DIR}/ext/bin" NO_DEFAULT_PATH)
    find_program(VERILATOR_PATH "verilator")
  endif()

  if(EXISTS ${VERILATOR_PATH})
    message(STATUS "Found Verilator at ${VERILATOR_PATH}.")

    # Find Verilator version.
    execute_process(COMMAND ${VERILATOR_PATH} --version
      OUTPUT_VARIABLE VERILATOR_VERSION)
    string(REGEX MATCH "Verilator (([0-9]+)\.([0-9]+)) \.*"
      MATCH ${VERILATOR_VERSION})
    # It's gotta be at least v4.110.
    if (${CMAKE_MATCH_1} LESS 4.110)
      message(FATAL_ERROR "CIRCT only supports Verilator version 4.110 and up. \
                           Found version: ${CMAKE_MATCH_1}. You can disable \
                           the Verilator tests with '-DVERILATOR_DISABLE=ON'.")
      set(VERILATOR_PATH "")
    endif()
  else()
    set(VERILATOR_PATH "")
    message(STATUS "Did not find Verilator.")
  endif()
endif()

#-------------------------------------------------------------------------------
# Vivado Configuration
#-------------------------------------------------------------------------------

# If vivado hasn't been explicitly disabled, find it.
option(VIVADO_DISABLE "Disable the vivado synthesis tests.")
if (VIVADO_DISABLE)
  message(STATUS "Disabling vivado tests.")
else()
  if (EXISTS ${VIVADO_PATH})
    get_filename_component(VIVADO_PATH ${VIVADO_PATH} DIRECTORY)
    message(STATUS "Setting vivado path to ${VIVADO_PATH}.")
  else()
    # Search for vivado's `vivado` command.
    find_program(VIVADO_PATH "vivado")
    if(EXISTS ${VIVADO_PATH})
      # Then strip the filename.
      get_filename_component(VIVADO_PATH ${VIVADO_PATH} DIRECTORY)
      message(STATUS "Found vivado at ${VIVADO_PATH}.")
    else()
      set(VIVADO_PATH "")
      message(STATUS "Did not find vivado.")
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# Clang-Tidy Configuration (for integration tests to check SystemC linting)
#-------------------------------------------------------------------------------

# If clang-tidy hasn't been explicitly disabled, find it.
option(CLANG_TIDY_DISABLE "Disable the clang-tidy lint tests.")
if (CLANG_TIDY_DISABLE)
  message(STATUS "Disabling clang-tidy lint tests.")
else()
  if (EXISTS ${CLANG_TIDY_PATH})
    get_filename_component(CLANG_TIDY_PATH ${CLANG_TIDY_PATH} DIRECTORY)
    message(STATUS "Setting clang-tidy path to ${CLANG_TIDY_PATH}.")
  else()
    # Search for the `clang-tidy` command.
    find_program(CLANG_TIDY_PATH "clang-tidy")
    if(EXISTS ${CLANG_TIDY_PATH})
      # Then strip the filename.
      get_filename_component(CLANG_TIDY_PATH ${CLANG_TIDY_PATH} DIRECTORY)
      message(STATUS "Found clang-tidy at ${CLANG_TIDY_PATH}.")
    else()
      set(CLANG_TIDY_PATH "")
      message(STATUS "Did NOT find clang-tidy.")
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# SystemC Configuration
#-------------------------------------------------------------------------------

# If SystemC hasn't been explicitly disabled, find it.
option(SYSTEMC_DISABLE "Disable the systemc tests.")
if (SYSTEMC_DISABLE)
  message(STATUS "Disabling systemc tests.")
else()
  find_file(HAVE_SYSTEMC systemc PATH /usr/include /usr/local/include ${SYSTEMC_PATH})
  if(HAVE_SYSTEMC)
    message(STATUS "Found systemc headers.")
  else()
    set(HAVE_SYSTEMC "")
    message(STATUS "Did NOT find systemc headers.")
  endif()
endif()

#-------------------------------------------------------------------------------
# Quartus Configuration
#-------------------------------------------------------------------------------

# If Quartus hasn't been explicitly disabled, find it.
option(QUARTUS_DISABLE "Disable the Quartus synthesis tests.")
if (QUARTUS_DISABLE)
  message(STATUS "Disabling Quartus tests.")
else()
  if (EXISTS ${QUARTUS_PATH})
    message(STATUS "Setting Quartus path to ${QUARTUS_PATH}.")
  else()
    # Search for Quartus's `quartus` command.
    find_program(QUARTUS_PATH "quartus")
    if(EXISTS ${QUARTUS_PATH})
      # Then strip the filename.
      get_filename_component(QUARTUS_PATH ${QUARTUS_PATH} DIRECTORY)
      message(STATUS "Found Quartus at ${QUARTUS_PATH}.")
    else()
      set(QUARTUS_PATH "")
      message(STATUS "Did not find Quartus.")
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# Questa Configuration
#-------------------------------------------------------------------------------

# If Questa hasn't been explicitly disabled, find it.
option(QUESTA_DISABLE "Disable the Questa simulation tests.")
if (QUESTA_DISABLE)
  message(STATUS "Disabling Questa tests.")
else()
  if (EXISTS ${QUESTA_PATH})
    message(STATUS "Setting Questa path to ${QUESTA_PATH}.")
  else()
    # Search for Questa's `vsim` command.
    find_program(QUESTA_PATH "vsim")
    if(EXISTS ${QUESTA_PATH})
      # Then strip the filename.
      get_filename_component(QUESTA_PATH ${QUESTA_PATH} DIRECTORY)
      message(STATUS "Found Questa at ${QUESTA_PATH}.")
    else()
      set(QUESTA_PATH "")
      message(STATUS "Did not find Questa.")
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# Yosys Configuration
#-------------------------------------------------------------------------------

# If Yosys hasn't been explicitly disabled, find it.
option(YOSYS_DISABLE "Disable the yosys tests.")
if (YOSYS_DISABLE)
  message(STATUS "Disabling yosys tests.")
else()
  find_program(YOSYS_PATH "yosys")
  if(EXISTS ${YOSYS_PATH})
    message(STATUS "Found yosys at ${YOSYS_PATH}.")
  else()
    set(YOSYS_PATH "")
    message(STATUS "Did not find yosys.")
  endif()
endif()

#-------------------------------------------------------------------------------
# Icarus Verilog Configuration
#-------------------------------------------------------------------------------

# If Icarus Verilog hasn't been explicitly disabled, find it.
option(IVERILOG_DISABLE "Disable the Icarus Verilog tests.")
if (IVERILOG_DISABLE)
  message(STATUS "Disabling Icarus Verilog tests.")
else()
  find_program(IVERILOG_PATH "iverilog")
  if(EXISTS ${IVERILOG_PATH})
    # Find iverilog version.
    execute_process(COMMAND ${IVERILOG_PATH} -V
      OUTPUT_VARIABLE IVERILOG_VERSION)

    string(REGEX MATCH "Icarus Verilog version (([0-9]+)\.([0-9]+)) \.*"
      MATCH ${IVERILOG_VERSION})

    if (${CMAKE_MATCH_1} LESS 11.0)
      message(FATAL_ERROR "CIRCT only supports Icarus Verilog version 11.0 and up. \
                           Found version: ${CMAKE_MATCH_1}. You can disable \
                           the Icarus Verilog tests with '-DIVERILOG_DISABLE=ON'.")
      set(IVERILOG_PATH "")
    endif()
    message(STATUS "Found iverilog at ${IVERILOG_PATH}.")
  else()
    set(IVERILOG_PATH "")
    message(STATUS "Did not find iverilog.")
  endif()
endif()

#-------------------------------------------------------------------------------
# OR-Tools Configuration
#-------------------------------------------------------------------------------

option(OR_TOOLS_DISABLE "Disable OR-Tools.")
if (OR_TOOLS_DISABLE)
  message(STATUS "Disabling OR-Tools.")
else()
  if(DEFINED OR_TOOLS_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${OR_TOOLS_PATH})
  endif()

  list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ext)

  find_package(ortools CONFIG)

  if (ortools_FOUND)
    get_filename_component(ortools_CMAKEDIR ${ortools_DIR} DIRECTORY)
    get_filename_component(ortools_LIBDIR ${ortools_CMAKEDIR} DIRECTORY)
    list(APPEND CMAKE_INSTALL_RPATH ${ortools_LIBDIR})
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  endif()
endif()

#-------------------------------------------------------------------------------
# Z3Lib Configuration (for circt-lec and SMT dialect integration tests)
#-------------------------------------------------------------------------------

# If z3 hasn't been explicitly disabled, find it.
option(Z3_DISABLE "Disable the z3 tests.")
if (Z3_DISABLE)
  message(STATUS "Disabling tests requiring z3.")
else()
  if(Z3_DIR)
    # Search and load the package configuration file in the specified directory.
    find_package(Z3 CONFIG REQUIRED PATHS ${Z3_DIR} NO_DEFAULT_PATH)
    if(Z3_FOUND)
      # Report the found library location and version
      # similarly to LLVM's `FindZ3` CMake module.
      get_target_property(Z3_LIB_LOCATION z3::libz3 IMPORTED_LOCATION_DEBUG)
      message(STATUS "Found Z3: ${Z3_LIB_LOCATION} (found version \"${Z3_VERSION_STRING}\")")
    endif()
  else()
    # Attempt initialising Z3 according to LLVM's `FindZ3` CMake module.
    find_package(Z3)
  endif()

  if(Z3_FOUND)
    SET(CIRCT_LEC_Z3_VER 4.8.11)
    if(Z3_VERSION_STRING VERSION_LESS ${CIRCT_LEC_Z3_VER})
      message(WARNING "Cannot build circt-lec with outdated Z3 version ${Z3_VERSION_STRING}, requires ${CIRCT_LEC_Z3_VER}.")
    else()
      message(STATUS "Z3 identified as a logical backend.")
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# Z3 Configuration (for integration tests to check SMT-LIB export)
#-------------------------------------------------------------------------------

# If z3 hasn't been explicitly disabled, find it.
option(Z3_DISABLE "Disable the z3 tests.")
if (Z3_DISABLE)
  message(STATUS "Disabling tests requiring z3.")
else()
  # Search for the `z3` command.
  find_program(Z3_PATH "z3")
  if(EXISTS ${Z3_PATH})
    # Then strip the filename.
    get_filename_component(Z3_PATH ${Z3_PATH} DIRECTORY)
    message(STATUS "Found z3 at ${Z3_PATH}.")
  else()
    set(Z3_PATH "")
    message(STATUS "Did NOT find z3.")
  endif()
endif()

#-------------------------------------------------------------------------------
# Python Configuration
#-------------------------------------------------------------------------------

option(CIRCT_BINDINGS_PYTHON_ENABLED "Enables CIRCT Python bindings." OFF)

if(CIRCT_BINDINGS_PYTHON_ENABLED)
  message(STATUS "CIRCT Python bindings are enabled.")
  set(MLIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES 0)
  mlir_detect_nanobind_install()
  # Prime the search like mlir_configure_python_dev_modules
  find_package(Python3 3.8 COMPONENTS Interpreter Development)
  find_package(Python3 3.8 COMPONENTS Interpreter Development.Module)
  find_package(nanobind 2.4.0 CONFIG REQUIRED)
else()
  message(STATUS "CIRCT Python bindings are disabled.")
  # Lookup python either way as some integration tests use python without the
  # bindings
  find_package(Python3)
  if(Python3_FOUND)
    message(STATUS "Found python at ${Python3_EXECUTABLE}")
  endif()
endif()

#-------------------------------------------------------------------------------
# Tcl bindings
#-------------------------------------------------------------------------------
option(CIRCT_BINDINGS_TCL_ENABLED "Enables CIRCT Tcl bindings." OFF)

llvm_canonicalize_cmake_booleans(CIRCT_BINDINGS_TCL_ENABLED)
if(CIRCT_BINDINGS_TCL_ENABLED)
  message(STATUS "CIRCT Tcl bindings are enabled")
  find_package(TCL 8.6 REQUIRED)
  find_package(TclStub 8.6 REQUIRED)
  message(STATUS "Found TCL include path: ${TCL_INCLUDE_PATH}")
  message(STATUS "Found TCL library: ${TCL_LIBRARY}")
  message(STATUS "Found TCL executable: ${TCL_TCLSH}")
endif()

#-------------------------------------------------------------------------------
# slang Verilog Frontend
#-------------------------------------------------------------------------------

option(CIRCT_SLANG_FRONTEND_ENABLED "Enables the slang Verilog frontend." OFF)
option(CIRCT_SLANG_BUILD_FROM_SOURCE
  "Build slang from source instead of finding an installed package" ON)

llvm_canonicalize_cmake_booleans(CIRCT_SLANG_FRONTEND_ENABLED)
llvm_canonicalize_cmake_booleans(CIRCT_SLANG_BUILD_FROM_SOURCE)

if(CIRCT_SLANG_FRONTEND_ENABLED)
  message(STATUS "slang Verilog frontend is enabled")
  if(CIRCT_SLANG_BUILD_FROM_SOURCE)
    # Build slang as part of CIRCT (see https://sv-lang.com/building.html)
    message(STATUS "Building slang from source")
    include(FetchContent)
    FetchContent_Declare(
      slang
      GIT_REPOSITORY https://github.com/MikePopoloski/slang.git
      GIT_TAG v3.0
      GIT_SHALLOW ON)
    set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE "NEVER")

    # Force Slang to be built as a static library to avoid messing around with
    # RPATHs and installing a slang dylib alongside CIRCT. The static library
    # will embed Slang into ImportVerilog.
    set(ORIGINAL_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(ORIGINAL_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

    if (MSVC)
      set(CMAKE_CXX_FLAGS "/EHsc")
    else ()
      set(CMAKE_CXX_FLAGS "")
    endif ()
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(slang)

    set(CMAKE_CXX_FLAGS ${ORIGINAL_CMAKE_CXX_FLAGS})
    set(BUILD_SHARED_LIBS ${ORIGINAL_BUILD_SHARED_LIBS})

    if(BUILD_SHARED_LIBS)
      set_target_properties(slang_slang PROPERTIES POSITION_INDEPENDENT_CODE ON)
      set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)
      set_target_properties(unordered_dense PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    # The following feels *very* hacky, but CMake complains about the
    # CIRCTImportVerilog target linking against slang_slang (even with PRIVATE
    # linking) without the latter being in an export set. I think we'd want to
    # statically link slang into the CIRCTImportVerilog library, but seems to be
    # harder than it ought to be.
    set_property(
      GLOBAL APPEND PROPERTY CIRCT_EXPORTS slang_slang unordered_dense fmt)
    
    # Disable the installation of headers coming from third-party libraries. We
    # won't use those APIs directly. Just make them static libraries for the sake
    # of running slang normally.
    set_target_properties(fmt PROPERTIES PUBLIC_HEADER "")
    set_target_properties(unordered_dense PROPERTIES PUBLIC_HEADER "")

    install(TARGETS slang_slang unordered_dense fmt EXPORT CIRCTTargets)
  else()
    find_package(slang 3.0 REQUIRED)
  endif()
endif()

#-------------------------------------------------------------------------------
# Arcilator JIT
#-------------------------------------------------------------------------------

if(MLIR_ENABLE_EXECUTION_ENGINE)
  set(ARCILATOR_JIT_ENABLED 1)
else()
  set(ARCILATOR_JIT_ENABLED 0)
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

if (CIRCT_INCLUDE_TESTS)
  add_definitions(-DCIRCT_INCLUDE_TESTS)
endif()

add_subdirectory(include/circt)
add_subdirectory(lib)
if(CIRCT_INCLUDE_TOOLS)
  add_subdirectory(tools)
endif()
add_subdirectory(frontends)

if (CIRCT_INCLUDE_TESTS)
  # Test directories are added after everything else so their cmake files can
  # use targets and target properties defined by the things they test.
  if (CIRCT_GTEST_AVAILABLE)
    add_subdirectory(unittests)
  endif()
  add_subdirectory(test)
  add_subdirectory(integration_test)
endif()

option(CIRCT_INCLUDE_DOCS "Generate build targets for the CIRCT docs.")
if (CIRCT_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

install(DIRECTORY include/circt include/circt-c
  DESTINATION include
  COMPONENT circt-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "*.sv"
  PATTERN "LICENSE.TXT"
  )

install(DIRECTORY ${CIRCT_INCLUDE_DIR}/circt ${CIRCT_INCLUDE_DIR}/circt-c
  DESTINATION include
  COMPONENT circt-headers
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
  add_llvm_install_targets(install-circt-headers
                           DEPENDS circt-headers
                           COMPONENT circt-headers)
endif()


# Custom target to install all CIRCT libraries
add_custom_target(circt-libraries)
set_target_properties(circt-libraries PROPERTIES FOLDER "Misc")

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-circt-libraries
                           DEPENDS circt-libraries
                           COMPONENT circt-libraries)
endif()

get_property(CIRCT_LIBS GLOBAL PROPERTY CIRCT_ALL_LIBS)
if(CIRCT_LIBS)
  list(REMOVE_DUPLICATES CIRCT_LIBS)
  foreach(lib ${CIRCT_LIBS})
    add_dependencies(circt-libraries ${lib})
    if(NOT LLVM_ENABLE_IDE)
      add_dependencies(install-circt-libraries install-${lib})
      add_dependencies(install-circt-libraries-stripped install-${lib}-stripped)
    endif()
  endforeach()
endif()

add_subdirectory(cmake/modules)

# Set RPATH to $ORIGIN on all targets.
function(set_rpath_all_targets dir)
  get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
  foreach(subdir ${subdirectories})
    set_rpath_all_targets(${subdir})
  endforeach()

  get_directory_property(LCL_TARGETS DIRECTORY ${dir} BUILDSYSTEM_TARGETS)
  set_property(TARGET ${LCL_TARGETS} PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
endfunction()

option(STANDALONE_INSTALL "Create an 'install' for packaging which doesn't \
         require installation" off)
if (STANDALONE_INSTALL)
  message(STATUS "Setting an $ORIGIN-based RPATH on all executables")
  set_rpath_all_targets(${CMAKE_CURRENT_SOURCE_DIR})
endif()
