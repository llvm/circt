# REQUIRES: bindings_tcl
# RUN: tclsh %s -- %TCL_PATH%
load [lindex $argv 1]libcirct-tcl[info sharedlibextension]

set circuit [circt load MLIR [lindex $argv 1]../../../../tcl/simple.mlir]
puts $circuit

# CHECK: hw.module.extern @ichi(%a: i2, %b: i3) -> (%c: i4, %d: i5)
# CHECK: 
# CHECK: hw.module @owo() -> (%owo_result : i32) {
# CHECK:   %0 = hw.constant 3 : i32
# CHECK:   hw.output %0 : i32
# CHECK: }
# CHECK: hw.module @uwu() -> () { }
# CHECK: 
# CHECK: hw.module @nya(%nya_input : i32) -> () {
# CHECK:   hw.instance "uwu1" @uwu() : () -> ()
# CHECK: }
# CHECK: 
# CHECK: hw.module @test() -> (%test_result : i32) {
# CHECK:   %myArray1 = sv.wire : !hw.inout<array<42 x i8>>
# CHECK:   %0 = hw.instance "owo1" @owo() : () -> (i32)
# CHECK:   hw.instance "nya1" @nya(%0) : (i32) -> ()
# CHECK:   hw.output %0 : i32
# CHECK: }
# CHECK: 
# CHECK: hw.module @always() -> () {
# CHECK:   %clock = hw.constant 1 : i1
# CHECK:   %3 = sv.wire : !hw.inout<i1>
# CHECK:   %false = hw.constant 0 : i1
# CHECK:   sv.alwaysff(posedge %clock)  {
# CHECK:     sv.passign %3, %false : i1
# CHECK:   }
# CHECK:   hw.output
# CHECK: }

