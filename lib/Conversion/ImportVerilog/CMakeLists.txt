# slang uses exceptions
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)

# For ABI compatibility, define the DEBUG macro in debug builds. Slang sets this
# internally. If we don't set this here as well, header-defined things like the
# destructor of `Driver`, which is generated in ImportVerilog's compilation
# unit, will destroy a different set of fields than what was potentially built
# or modified by code compiled in the Slang compilation unit.
add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)

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
endif ()

add_circt_translation_library(CIRCTImportVerilog
  Expressions.cpp
  FormatStrings.cpp
  HierarchicalNames.cpp
  ImportVerilog.cpp
  Statements.cpp
  Structure.cpp
  Types.cpp
  TimingControls.cpp

  DEPENDS
  slang_slang

  LINK_LIBS PUBLIC
  CIRCTDebug
  CIRCTHW
  CIRCTMoore
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRTranslateLib
  PRIVATE
  slang_slang
)
