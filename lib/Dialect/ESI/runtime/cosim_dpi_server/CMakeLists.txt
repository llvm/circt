##===- CMakeLists.txt - Core cosim DPI library ----------------*- cmake -*-===//
##
## Define the cosim DPI library if it's enabled.
##
##===----------------------------------------------------------------------===//

# Dummy library for a library which should be included by the RTL simulator.
# Dummy is necessary for linking purposes.
add_library(MtiPli SHARED
  DummySvDpi.cpp
)
set_target_properties(MtiPli PROPERTIES CXX_VISIBILITY_PRESET "default")
# Technically, we shouldn't have to install this as is _should_ be provided by
# the simulator, but without it we get runtime link errors. TODO: figure out how
# to do this properly such that we avoid packaging a dummy library.
install(TARGETS MtiPli
  DESTINATION ${ESIRT_INSTALL_LIBDIR}
  COMPONENT ESIRuntime
)

# DPI calls.
add_library(EsiCosimDpiServer SHARED
  DpiEntryPoints.cpp
)
set_target_properties(EsiCosimDpiServer
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        CXX_VISIBILITY_PRESET "default"
)
add_dependencies(EsiCosimDpiServer ESICppRuntime MtiPli)
target_link_libraries(EsiCosimDpiServer
  PUBLIC
    ESICppRuntime
    CosimBackend
    MtiPli
)

add_dependencies(ESIRuntime EsiCosimDpiServer)
install(TARGETS EsiCosimDpiServer
  DESTINATION ${ESIRT_INSTALL_LIBDIR}
  COMPONENT ESIRuntime
)

# RTL cosimulation collateral.
set(cosim_collateral
  Cosim_DpiPkg.sv
  Cosim_Endpoint.sv
  Cosim_Manifest.sv

  driver.sv
  driver.cpp
)

install(FILES
  ${cosim_collateral}
  DESTINATION esiaccel/cosim
  COMPONENT ESIRuntime
)

add_custom_target(esi-cosim
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/esi-cosim.py
          ${CMAKE_BINARY_DIR}/bin/esi-cosim.py)
foreach (cf ${cosim_collateral})
  add_custom_command(TARGET esi-cosim POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_CURRENT_SOURCE_DIR}/${cf}
              ${CMAKE_CURRENT_BINARY_DIR}/../python/esiaccel/cosim/${cf}
  )
endforeach()

# ESI simple cosim runner.
install(FILES
  esi-cosim.py
  DESTINATION ${ESIRT_INSTALL_BINDIR}
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
              GROUP_EXECUTE GROUP_READ
              WORLD_EXECUTE WORLD_READ
  COMPONENT ESIRuntime
)
add_dependencies(ESIRuntime esi-cosim)
set(ESI_COSIM_PATH $<TARGET_FILE:EsiCosimDpiServer>
      CACHE PATH "Path to Cosim DPI shared library")
