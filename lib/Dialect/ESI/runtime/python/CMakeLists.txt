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

# Pybind11 is used to wrap the ESICppRuntime APIs.
find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
  IF(MSVC)
    # Work around an issue with pybind11 and cmake incompatibility on Windows in debug mode.
    set_target_properties(Python3::Module PROPERTIES
          MAP_IMPORTED_CONFIG_DEBUG ";RELEASE")
  ENDIF(MSVC)

  if(pybind11_DIR)
    message(STATUS "Using explicit pybind11 cmake directory: ${pybind11_DIR} (-Dpybind11_DIR to change)")
  else()
    message(STATUS "Checking for pybind11 in python path...")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}"
      -c "import pybind11;print(pybind11.get_cmake_dir(), end='')"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE STATUS
      OUTPUT_VARIABLE PACKAGE_DIR
      ERROR_QUIET)
    if(NOT STATUS EQUAL "0")
      message(FATAL_ERROR "pybind11 not found (install via 'pip install pybind11' or set pybind11_DIR)")
    endif()
    message(STATUS "found (${PACKAGE_DIR})")
    set(pybind11_DIR "${PACKAGE_DIR}")
  endif()

  # Now, find pybind11.
  find_package(pybind11 CONFIG)
  if (NOT pybind11_FOUND)
    message (STATUS "Could not find pybind11. Disabling Python API.")
    if (WHEEL_BUILD)
      message (FATAL_ERROR "pybind11 is required for a wheel build.")
    endif()
  else()
    # Compile Pybind11 module and copy to the correct python directory.
    pybind11_add_module(esiCppAccel
      ${CMAKE_CURRENT_SOURCE_DIR}/esiaccel/esiCppAccel.cpp)
    target_link_libraries(esiCppAccel PRIVATE ESICppRuntime)
    set_target_properties(esiCppAccel PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/esiaccel"
    )

    # Check for stubgen and generate stubs if available.
    find_program(STUBGEN pybind11-stubgen)
    if ("${STUBGEN}" STREQUAL "STUBGEN-NOTFOUND")
      message(STATUS "pybind11_stubgen not found. Skipping stub generation.")
    else()
      if(WIN32)
        # I just wasted all day trying to figure out the DLL search path on
        # Windows both locally and in the runner. I'm done. Windows wheels
        # won't have a stub until somebody else figures this out.
        # TODO: have the patience to make this work.
        message(WARNING "pybind11-stubgen is not supported on Windows.")
      else()
        set(stubgen_python_path "$ENV{PYTHONPATH}:${CMAKE_CURRENT_BINARY_DIR}")
        add_custom_command(
            TARGET esiCppAccel
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH="${stubgen_python_path}"
                ${STUBGEN}
                  -o "${CMAKE_CURRENT_BINARY_DIR}"
                  esiaccel.esiCppAccel
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
    endif()

    if (WHEEL_BUILD)
      if ("${STUBGEN}" STREQUAL "STUBGEN-NOTFOUND")
        message (FATAL_ERROR "pybind11_stubgen is required for a wheel build.")
      endif()
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
    # and Pybind11 module.
    add_custom_target(ESIPythonRuntime
      DEPENDS
        ${ESIPythonRuntimeSources}
        esiCppAccel
    )

    add_dependencies(ESIRuntime ESIPythonRuntime)

  endif()
else() # Python not found.
  if (WHEEL_BUILD)
    message (FATAL_ERROR "python-dev is required for a wheel build.")
  endif()
endif()
