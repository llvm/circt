set(ArcRuntimeLibSources
  ArcRuntime.cpp
  ModelInstance.cpp
  TraceEncoder.cpp
  VCDTraceEncoder.cpp
)

set(LLVM_OPTIONAL_SOURCES
  FSTTraceEncoder.cpp
  )

if(CIRCT_LIBFST_ENABLED)
  list(APPEND ArcRuntimeLibSources FSTTraceEncoder.cpp)
endif()

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

if(CIRCT_LIBFST_ENABLED)
  target_compile_definitions(CIRCTArcRuntime PUBLIC CIRCT_LIBFST_ENABLED=1)
  target_link_libraries(CIRCTArcRuntime PRIVATE fst)
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
if(CIRCT_LIBFST_ENABLED)
  target_compile_definitions(obj.CIRCTArcJITRuntime PUBLIC CIRCT_LIBFST_ENABLED=1)
  target_link_libraries(CIRCTArcJITRuntime PRIVATE fst)
endif()
