# slang uses exceptions
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)

# Disable some compiler warnings caused by slang headers such that the
# `ImportVerilog` build doesn't spew out a ton of warnings that are not related
# to CIRCT.
if (MSVC)
  # No idea what to put here
else ()
  # slang uses exceptions; we intercept these in ImportVerilog
  add_compile_options(-fexceptions)
  add_compile_options(-frtti)
  # slang has some classes with virtual funcs but non-virtual destructor.
  add_compile_options(-Wno-non-virtual-dtor)
  # some other warnings we've seen
  add_compile_options(-Wno-c++98-compat-extra-semi)
  add_compile_options(-Wno-ctad-maybe-unsupported)
  add_compile_options(-Wno-cast-qual)
  # visitor switch statements cover all cases but have default
  add_compile_options(-Wno-covered-switch-default)
  # several macro flags slang used
  add_compile_definitions(DEBUG)
  add_compile_definitions(BOOST_ALL_NO_LIB)
  add_compile_definitions(SLANG_STATIC_DEFINE)
  add_compile_definitions(SLANG_USE_BOOST)

  # compiler version check needed in slang 3.0+
  if(slang_VERSION VERSION_GREATER 3.0)
    # check Clang version
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16)
      message(FATAL_ERROR "Clang version must be >= 16")
    endif()
    # check GCC version
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10)
      message(FATAL_ERROR "GCC version must be >= 10")
    endif()
  endif()
endif ()

set(CMAKE_CXX_STANDARD 20)
add_circt_translation_library(CIRCTImportVerilog
  ImportVerilog.cpp
  Structure.cpp
  Types.cpp

  LINK_LIBS PUBLIC
  CIRCTMoore
  MLIRTranslateLib
  PRIVATE
  slang::slang
)
