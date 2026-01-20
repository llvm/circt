##===----------------------------------------------------------------------===//
## Python bindings for the ESI runtime.
##===----------------------------------------------------------------------===//

set(ESIPythonRuntimeSources
  esiaccel/__init__.py
  esiaccel/accelerator.py
  esiaccel/codegen.py
  esiaccel/types.py
  esiaccel/utils.py
  esiaccel/cosim/questa.py
  esiaccel/cosim/simulator.py
  esiaccel/cosim/verilator.py
)

# Nanobind is used to wrap the ESICppRuntime APIs.
find_package(Python3 COMPONENTS Interpreter Development.Module)
if(Python3_FOUND)
  IF(MSVC)
    # Work around an issue with nanobind and cmake incompatibility on Windows in debug mode.
    set_target_properties(Python3::Module PROPERTIES
          MAP_IMPORTED_CONFIG_DEBUG ";RELEASE")
  ENDIF(MSVC)

  if(nanobind_DIR)
    message(STATUS "Using explicit nanobind cmake directory: ${nanobind_DIR} (-Dnanobind_DIR to change)")
  else()
    message(STATUS "Checking for nanobind in python path...")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}"
      -c "import nanobind;print(nanobind.cmake_dir(), end='')"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE STATUS
      OUTPUT_VARIABLE PACKAGE_DIR
      ERROR_QUIET)
    if(NOT STATUS EQUAL "0")
      message(FATAL_ERROR "nanobind not found (install via 'pip install nanobind' or set nanobind_DIR)")
    endif()
    message(STATUS "found (${PACKAGE_DIR})")
    set(nanobind_DIR "${PACKAGE_DIR}")
  endif()

  # Now, find nanobind.
  find_package(Python COMPONENTS Interpreter Development.Module)
  find_package(nanobind CONFIG)
  if (NOT nanobind_FOUND)
    message (STATUS "Could not find nanobind. Disabling Python API.")
    if (WHEEL_BUILD)
      message (FATAL_ERROR "nanobind is required for a wheel build.")
    endif()
  else()
    # Compile nanobind module and copy to the correct python directory.
    nanobind_add_module(esiCppAccel
      ${CMAKE_CURRENT_SOURCE_DIR}/esiaccel/esiCppAccel.cpp)

    # For wheel builds on Linux, force static linking of libstdc++ to avoid
    # undefined symbols when ESICppRuntime's static libstdc++ symbols aren't
    # exported in its dynamic symbol table.
    if(WHEEL_BUILD AND UNIX AND NOT APPLE)
      target_link_options(esiCppAccel PRIVATE
        -static-libstdc++
        -static-libgcc
      )
    endif()

    target_link_libraries(esiCppAccel PRIVATE ESICppRuntime)
    set_target_properties(esiCppAccel PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/esiaccel"
    )

    # Use nanobind's built-in stubgen for stub generation.
    if(WIN32)
      # I just wasted all day trying to figure out the DLL search path on
      # Windows both locally and in the runner. I'm done. Windows wheels
      # won't have a stub until somebody else figures this out.
      # TODO: have the patience to make this work.
      message(WARNING "Stub generation is not supported on Windows.")
    else()
      set(stubgen_python_path "$ENV{PYTHONPATH}:${CMAKE_CURRENT_BINARY_DIR}")
      add_custom_command(
          TARGET esiCppAccel
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH="${stubgen_python_path}"
              "${Python3_EXECUTABLE}" -m nanobind.stubgen
                -m esiaccel.esiCppAccel
                -o "${CMAKE_CURRENT_BINARY_DIR}/esiaccel/esiCppAccel.pyi"
      )
      if (WHEEL_BUILD)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/esiaccel/esiCppAccel.pyi"
          DESTINATION .
          COMPONENT ESIRuntime
        )
      else()
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/esiaccel/esiCppAccel.pyi"
          DESTINATION python/esiaccel
          COMPONENT ESIRuntime
        )
      endif()
    endif()

    if (WHEEL_BUILD)
      set_target_properties(esiCppAccel PROPERTIES
        INSTALL_RPATH "$ORIGIN/lib")
    else()
      set_target_properties(esiCppAccel PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../lib")
    endif()
    set_target_properties(esiCppAccel PROPERTIES
      INSTALL_RPATH_USE_LINK_PATH FALSE)

    if (WHEEL_BUILD)
      install(TARGETS esiCppAccel
        DESTINATION .
        COMPONENT ESIRuntime
      )
    else()
      install(TARGETS esiCppAccel
        DESTINATION python/esiaccel
        COMPONENT ESIRuntime
      )
    endif()
    install(RUNTIME_DEPENDENCY_SET ESICppRuntime_RUNTIME_DEPS
      DESTINATION ${ESIRT_INSTALL_LIBDIR}
      PRE_EXCLUDE_REGEXES .*
      COMPONENT ESIRuntime
    )

    foreach(pysrc ${ESIPythonRuntimeSources})
      # Copy each of the Python sources to the build dir.
      add_custom_command(
          OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${pysrc}
          COMMAND ${CMAKE_COMMAND} -E copy
              ${CMAKE_CURRENT_SOURCE_DIR}/${pysrc}
              ${CMAKE_CURRENT_BINARY_DIR}/${pysrc}
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${pysrc}
      )

      # Specify the install location for the Python sources.
      get_filename_component(DEST ${pysrc} DIRECTORY)
      # If we're building a wheel, we need to install to the root directory.
      if (WHEEL_BUILD)
        string(REPLACE "esiaccel" "." DEST ${DEST})
      endif()

      install(FILES ${pysrc}
        DESTINATION ${DEST}
        COMPONENT ESIRuntime)
    endforeach()

    # Custom target for the Python runtime just aggregates the python sources
    # and nanobind module.
    add_custom_target(ESIPythonRuntime
      DEPENDS
        ${ESIPythonRuntimeSources}
        esiCppAccel
    )

    add_dependencies(ESIRuntime ESIPythonRuntime)

  endif()
else() # Python not found.
  message(WARNING "Python3 not found. Disabling ESI Runtime Python API.")
  if (WHEEL_BUILD)
    message (FATAL_ERROR "python-dev is required for a wheel build.")
  endif()
endif()
