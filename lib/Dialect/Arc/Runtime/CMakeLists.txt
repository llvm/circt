set(ArcRuntimeLibSources
  ArcRuntime.cpp
  ModelInstance.cpp
  TraceEncoder.cpp
  VCDTraceEncoder.cpp
)

# Shipped version of the runtime for ahead-of-time compiled models
if (WIN32)
# Ship a DLL for Windows to avoid trouble with Debug/Release builds of
# the C runtime.
  add_circt_library(CIRCTArcRuntime
    SHARED
    EXCLUDE_FROM_LIBMLIR
    ${ArcRuntimeLibSources}
  )
else()
  add_circt_library(CIRCTArcRuntime
    EXCLUDE_FROM_LIBMLIR
    ${ArcRuntimeLibSources}
   )
endif()

# Install required header files
install(FILES
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Arc/Runtime/ArcRuntime.h
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Arc/Runtime/Common.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/circt/Dialect/Arc/Runtime
  COMPONENT CIRCTArcRuntime
)

# JIT runtime linked into the arcilator tool
add_circt_library(CIRCTArcJITRuntime
  EXCLUDE_FROM_LIBMLIR
  OBJECT
  ${ArcRuntimeLibSources}
)
target_compile_definitions(obj.CIRCTArcJITRuntime PUBLIC ARC_RUNTIME_JIT_BIND)
