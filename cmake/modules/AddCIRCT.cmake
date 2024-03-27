include_guard()

function(add_circt_dialect dialect dialect_namespace)
  add_mlir_dialect(${ARGV})
  add_dependencies(circt-headers MLIR${dialect}IncGen)
endfunction()

function(add_circt_interface interface)
  add_mlir_interface(${ARGV})
  add_dependencies(circt-headers MLIR${interface}IncGen)
endfunction()

# Additional parameters are forwarded to tablegen.
function(add_circt_doc tablegen_file output_path command)
  set(LLVM_TARGET_DEFINITIONS ${tablegen_file}.td)
  string(MAKE_C_IDENTIFIER ${output_path} output_id)
  tablegen(MLIR ${output_id}.md ${command} ${ARGN})
  set(GEN_DOC_FILE ${CIRCT_BINARY_DIR}/docs/${output_path}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md)
  add_custom_target(${output_id}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(circt-doc ${output_id}DocGen)
endfunction()

function(add_circt_dialect_doc dialect dialect_namespace)
  add_circt_doc(
    ${dialect} Dialects/${dialect}
    -gen-dialect-doc -dialect ${dialect_namespace})
endfunction()

function(add_circt_library name)
  add_mlir_library(${ARGV} DISABLE_INSTALL)
  add_circt_library_install(${name})
endfunction()

macro(add_circt_executable name)
  add_llvm_executable(${name} ${ARGN})
  set_target_properties(${name} PROPERTIES FOLDER "circt executables")
endmacro()

macro(add_circt_tool name)
  if (NOT CIRCT_BUILD_TOOLS)
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_circt_executable(${name} ${ARGN})

  if (CIRCT_BUILD_TOOLS)
    get_target_export_arg(${name} CIRCT export_to_circttargets)
    install(TARGETS ${name}
      ${export_to_circttargets}
      RUNTIME DESTINATION "${CIRCT_TOOLS_INSTALL_DIR}"
      COMPONENT ${name})

    if(NOT CMAKE_CONFIGURATION_TYPES)
      add_llvm_install_targets(install-${name}
        DEPENDS ${name}
        COMPONENT ${name})
    endif()
    set_property(GLOBAL APPEND PROPERTY CIRCT_EXPORTS ${name})
  endif()
endmacro()

# Adds a CIRCT library target for installation.  This should normally only be
# called from add_circt_library().
function(add_circt_library_install name)
  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  get_target_export_arg(${name} CIRCT export_to_circttargets UMBRELLA circt-libraries)
  install(TARGETS ${name}
    COMPONENT ${name}
    ${export_to_circttargets}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    # Note that CMake will create a directory like:
    #   objects-${CMAKE_BUILD_TYPE}/obj.LibName
    # and put object files there.
    OBJECTS DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  )

  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-${name}
                            DEPENDS ${name}
                            COMPONENT ${name})
  endif()
  set_property(GLOBAL APPEND PROPERTY CIRCT_ALL_LIBS ${name})
  endif()   
  set_property(GLOBAL APPEND PROPERTY CIRCT_EXPORTS ${name})
endfunction()

function(add_circt_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY CIRCT_DIALECT_LIBS ${name})
  add_circt_library(${ARGV} DEPENDS circt-headers)
endfunction()

function(add_circt_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY CIRCT_CONVERSION_LIBS ${name})
  add_circt_library(${ARGV} DEPENDS circt-headers)
endfunction()

function(add_circt_translation_library name)
  set_property(GLOBAL APPEND PROPERTY CIRCT_TRANSLATION_LIBS ${name})
  add_circt_library(${ARGV} DEPENDS circt-headers)
endfunction()

function(add_circt_verification_library name)
  set_property(GLOBAL APPEND PROPERTY CIRCT_VERIFICATION_LIBS ${name})
  add_circt_library(${ARGV} DEPENDS circt-headers)
endfunction()
