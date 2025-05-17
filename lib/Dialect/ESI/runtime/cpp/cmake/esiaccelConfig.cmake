add_library(esiaccel::ESICppRuntime SHARED IMPORTED)
set_target_properties(esiaccel::ESICppRuntime PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../include"
)

if(WIN32)
  set_target_properties(esiaccel::ESICppRuntime PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/../ESICppRuntime.dll"
    IMPORTED_IMPLIB "${CMAKE_CURRENT_LIST_DIR}/../ESICppRuntime.lib"
  )
else()
  set_target_properties(esiaccel::ESICppRuntime PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/../python/esiaccel/libESICppRuntime.so"
  )
endif()
