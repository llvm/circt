##===- CMakeLists.txt - Core cosim DPI library ----------------*- cmake -*-===//
##
## Define the cosim DPI library if it's enabled.
##
##===----------------------------------------------------------------------===//

add_library(EsiCosimDpiServer SHARED
  DpiEntryPoints.cpp
  Server.cpp
  Endpoint.cpp
)
set_target_properties(EsiCosimDpiServer
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        CXX_VISIBILITY_PRESET "default"
)
add_dependencies(EsiCosimDpiServer EsiCosimCapnp MtiPli)
target_link_libraries(EsiCosimDpiServer PRIVATE EsiCosimCapnp MtiPli)

set(ESI_COSIM_PATH $<TARGET_FILE:EsiCosimDpiServer>
      CACHE PATH "Path to Cosim DPI shared library")
