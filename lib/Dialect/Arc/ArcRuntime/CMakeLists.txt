set(ArcRuntimeLibSources
  ArcRuntime.cpp
  ModelInstance.cpp
)

# Shipped version of the runtime for ahead-of-time compiled models
if (WIN32)
# Ship a DLL for Windows to avoid trouble with Debug/Release builds of
# the C runtime.
  add_library(ArcRuntime SHARED ${ArcRuntimeLibSources})
else()
  add_library(ArcRuntime STATIC ${ArcRuntimeLibSources})
endif()
target_include_directories(ArcRuntime PUBLIC ${ARC_RUNTIME_INCLUDE})


# JIT runtime statically linked into the arcilator tool
add_library(ArcJITRuntime STATIC ${ArcRuntimeLibSources})
target_include_directories(ArcJITRuntime PUBLIC ${ARC_RUNTIME_INCLUDE})
target_compile_definitions(ArcJITRuntime PUBLIC ARC_RUNTIME_JIT_BIND)
