if (MSVC)
else ()
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
