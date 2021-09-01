include_guard()

function(add_circt_dialect dialect dialect_namespace)
  add_mlir_dialect(${ARGV})
  add_dependencies(circt-headers MLIR${dialect}IncGen)
endfunction()

function(add_circt_interface interface)
  add_mlir_interface(${ARGV})
  add_dependencies(circt-headers MLIR${interface}IncGen)
endfunction()

function(add_circt_doc doc_filename command output_file output_directory)
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  tablegen(MLIR ${output_file}.md ${command})
  set(GEN_DOC_FILE ${CIRCT_BINARY_DIR}/docs/${output_directory}${output_file}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(circt-doc ${output_file}DocGen)
endfunction()

function(add_circt_library name)
  add_mlir_library(${ARGV})
  add_circt_library_install(${name})
endfunction()

# Adds a CIRCT library target for installation.  This should normally only be
# called from add_circt_library().
function(add_circt_library_install name)
  set_property(GLOBAL APPEND PROPERTY CIRCT_ALL_LIBS ${name})
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

function(add_circt_analysis_library name)
  set_property(GLOBAL APPEND PROPERTY CIRCT_ANALYSIS_LIBS ${name})
  add_circt_library(${ARGV} DEPENDS circt-headers)
endfunction()
