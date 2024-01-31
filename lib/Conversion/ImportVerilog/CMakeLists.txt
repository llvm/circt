set(CMAKE_CXX_STANDARD 20)

# For ABI compatibility, define the DEBUG macro in debug builds. Slang sets this
# internally. If we don't set this here as well, header-defined things like the
# destructor of `Driver`, which is generated in ImportVerilog's compilation
# unit, will destroy a different set of fields than what was potentially built
# or modified by code compiled in the Slang compilation unit.
add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)

# If Slang is using its vendored version of `boost_unordered`, the `SLANG_BOOST_SINGLE_HEADER` define must be set.
if(TARGET boost_unordered)
  add_compile_definitions(SLANG_BOOST_SINGLE_HEADER)
endif()

# Disable some compiler warnings caused by slang headers such that the
# `ImportVerilog` build doesn't spew out a ton of warnings that are not related
# to CIRCT.
if (MSVC)
  # No idea what to put here
else ()
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
  ImportVerilog.cpp

  LINK_LIBS PUBLIC
  MLIRTranslateLib
  PRIVATE
  slang_slang
)
