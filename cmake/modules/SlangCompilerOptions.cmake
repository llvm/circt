# The Slang headers contain C++20 code. So anything that interfaces directly
# with Slang must be built accordingly.
set(CMAKE_CXX_STANDARD 20)

# For ABI compatibility, define the DEBUG macro in debug builds. Slang sets this
# internally. If we don't set this here as well, header-defined things like the
# destructor of `Driver`, which is generated in ImportVerilog's compilation
# unit, will destroy a different set of fields than what was potentially built
# or modified by code compiled in the Slang compilation unit.
add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)

# HACK: When the `OBJECT` argument is passed to `llvm_add_library()`,
# `COMPILE_DEFINITIONS` are not correctly inherited. For that reason, we
# manually set it here.
if(TARGET Boost::headers)
  add_compile_definitions(
    $<TARGET_PROPERTY:Boost::headers,INTERFACE_COMPILE_DEFINITIONS>)
endif()

# Disable some compiler warnings caused by slang headers such that the
# `ImportVerilog` build doesn't spew out a ton of warnings that are not related
# to CIRCT.
if (NOT MSVC)
  # slang has some classes with virtual funcs but non-virtual destructor.
  add_compile_options(-Wno-non-virtual-dtor)
  # some other warnings we've seen
  add_compile_options(-Wno-c++98-compat-extra-semi)
  add_compile_options(-Wno-ctad-maybe-unsupported)
  # visitor switch statements cover all cases but have default
  add_compile_options(-Wno-covered-switch-default)
endif ()
