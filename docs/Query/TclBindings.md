# Tcl Bindings for the Query API
## Building
To build the Tcl bindings for the Query API, you must use the flag `-DCIRCT_BINDINGS_TCL_ENABLED=ON`.

Afterwards, use ninja check-circt-integration to ensure that the bindings work. (This will now additionally spin up a couple of Tcl scripts to test that the bindings work.)

## Usage
The dynamic library is stored in `build/lib/Bindings/Tcl/` and is named `libcirct-tcl.[dylib|so|dll]`. Loading this file in a Tcl file gives you access to the `circt` procedure and a plethora of filters for querying. The following should work once you've compiled CIRCT with Tcl bindings:
```tcl
$ tclsh
% load build/lib/Bindings/Tcl/libcirct-tcl.so
% puts [circt load MLIR test.mlir]
module {
  hw.module.extern @ichi(%a: i2, %b: i3) -> (%c: i4, %d: i5)

  hw.module @owo() -> (%owo_result : i32) {
    %0 = hw.constant 3 : i32
    hw.output %0 : i32
  }
  hw.module @uwu() -> () { }

  hw.module @nya(%nya_input : i32) -> () {
    hw.instance "uwu1" @uwu() : () -> ()
  }

  hw.module @test() -> (%test_result : i32) {
    %myArray1 = sv.wire : !hw.inout<array<42 x i8>>
    %0 = hw.instance "owo1" @owo() : () -> (i32)
    hw.instance "nya1" @nya(%0) : (i32) -> ()
    hw.output %0 : i32
  }

  hw.module @always() -> () {
    %clock = hw.constant 1 : i1
    %3 = sv.wire : !hw.inout<i1>
    %false = hw.constant 0 : i1
    sv.alwaysff(posedge %clock)  {
      sv.passign %3, %false : i1
    }
    hw.output
  }
}
```

### The `circt` command
The `circt` procedure is how you perform actions on MLIR and FIR. The following commands are supported:
 - `circt load <MLIR|FIR> [file]`

    Loads an MLIR/FIR file (loading FIR files is currently unimplemented). Returns an operation.
 - `circt query [filter] [operation | list of ops]`

    Performs a query on an operation or a list of ops. Returns a list of operations.
 - `circt get modname [operation]`

    Gets the name of the module passed in.
 - `circt get opname [operation]`

    Gets the name of the operation passed in.
 - `circt get attrs [operation | list of ops] [attribute names]*`

    Gets the attributes of the given operations that match the list of names. If no names are provided, then all attributes are dumped. Returns a dictionary mapping operations to attributes.

### Filters
TODO

