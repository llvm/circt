function(add_esiaccel_flags TARGET)
  message("-- Adding ESI runtime flags to target ${TARGET}")
  target_include_directories(${TARGET}
    PUBLIC
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../include"
  )
  file(GLOB ESI_LIBS ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../lib/*)
  target_link_libraries(${TARGET}
    PUBLIC
      ${ESI_LIBS}
  )
endfunction()
