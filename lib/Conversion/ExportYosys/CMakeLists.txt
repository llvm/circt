# For ABI compatibility, define the DEBUG macro in debug builds. Slang sets this
# internally. If we don't set this here as well, header-defined things like the
# destructor of `Driver`, which is generated in ImportVerilog's compilation
# unit, will destroy a different set of fields than what was potentially built
# or modified by code compiled in the Slang compilation unit.
#add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)

# Disable some compiler warnings caused by slang headers such that the
# `ImportVerilog` build doesn't spew out a ton of warnings that are not related
# to CIRCT.
if (MSVC)
  # No idea what to put here
else ()
  # slang uses exceptions; we intercept these in ImportVerilog
  add_compile_options(-D_YOSYS_)
  add_compile_options(-DYOSYS_ENABLE_ABC)
  add_compile_options(-DYOSYS_ENABLE_GLOB)
  add_compile_options(-DYOSYS_ENABLE_PLUGINS)
  add_compile_options(-DYOSYS_ENABLE_READLINE)
  add_compile_options(-DYOSYS_ENABLE_COVER)
  add_compile_options(-DYOSYS_ENABLE_ZLIB)
  add_compile_options(-fexceptions)
  add_compile_options(-Wno-c++98-compat-extra-semi)
  add_compile_options(-Wno-cast-qual)
endif ()

add_circt_translation_library(CIRCTExportYosys
  ExportRTLIL.cpp
  ImportRTLIL.cpp
  YosysOptimizer.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTranslateLib
  MLIRIR
  MLIRTransforms
  CIRCTHW
  CIRCTSV
  CIRCTOM
  CIRCTComb
  libyosys
)
