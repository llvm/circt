set(ArcRuntimeLibSources
  ArcRuntime.cpp
  ModelInstance.cpp
)

# Shipped version of the runtime for ahead-of-time compiled models
if (WIN32)
# Ship a DLL for Windows to avoid trouble with Debug/Release builds of
# the C runtime.
  add_circt_library(CIRCTArcRuntime SHARED EXCLUDE_FROM_LIBMLIR ${ArcRuntimeLibSources})
else()
  add_circt_library(CIRCTArcRuntime EXCLUDE_FROM_LIBMLIR ${ArcRuntimeLibSources})
endif()

# JIT runtime statically linked into the arcilator tool
add_circt_library(CIRCTArcJITRuntime
  EXCLUDE_FROM_LIBMLIR
  OBJECT
  ${ArcRuntimeLibSources}
)
target_compile_definitions(obj.CIRCTArcJITRuntime PUBLIC ARC_RUNTIME_JIT_BIND)
