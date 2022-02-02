// RUN: circt-opt %s -split-input-file -verify-diagnostics

firrtl.circuit "X" {

firrtl.module @X(in %b : !firrtl.unknowntype) {
  // expected-error @-1 {{unknown FIRRTL dialect type: "unknowntype"}}
}

}

// -----

firrtl.circuit "X" {

firrtl.module @X(in %b : !firrtl.uint<32>, in %d : !firrtl.uint<16>, in %out : !firrtl.uint) {
  // expected-error @+1 {{'firrtl.add' op expected 2 operands, but found 3}}
  %3 = "firrtl.add"(%b, %d, %out) : (!firrtl.uint<32>, !firrtl.uint<16>, !firrtl.uint) -> !firrtl.uint<32>
}

}

// -----

// expected-error @+1 {{'firrtl.circuit' op must contain one module that matches main name 'MyCircuit'}}
firrtl.circuit "MyCircuit" {

"firrtl.module"() ( {
}) { type = () -> ()} : () -> ()

}

// -----


// expected-error @+1 {{'firrtl.module' op expects parent op 'firrtl.circuit'}}
firrtl.module @X() {}

// -----

// expected-error @+1 {{'firrtl.circuit' op must contain one module that matches main name 'Foo'}}
firrtl.circuit "Foo" {

firrtl.module @Bar() {}

}

// -----

// expected-error @+1 {{'firrtl.circuit' op must have a non-empty name}}
firrtl.circuit "" {
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{ports should all be FIRRTL types}}
firrtl.module @foo(in %a: i1) {}
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{requires 1 port directions}}
firrtl.module @foo(in %a : !firrtl.uint<1>) attributes {portDirections = 3 : i2} {}
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{requires 1 port names}}
firrtl.module @foo(in %a : !firrtl.uint<1>) attributes {portNames=[]} {}
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{port names should all be string attributes}}
firrtl.module @foo(in %a : !firrtl.uint<1>) attributes {portNames=[1 : i1]} {}
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{op requires 1 port annotations}}
firrtl.module @foo(in %a : !firrtl.uint<1>) attributes {portAnnotations=[[], []]} {}
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{annotations must be dictionaries or subannotations}}
firrtl.module @foo(in %a: !firrtl.uint<1> ["hello"]) {}
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{requires one region}}
"firrtl.module"() ( { }, { })
   {sym_name = "foo", portTypes = [!firrtl.uint], portDirections = 1 : i1,
    portNames = ["in0"], portAnnotations = [], portSyms = [""]} : () -> ()
}


// -----

firrtl.circuit "foo" {
// expected-error @+1 {{entry block must have 1 arguments to match module signature}}
"firrtl.module"() ( {
  ^entry:
}) {sym_name = "foo", portTypes = [!firrtl.uint], portDirections = 1 : i1,
    portNames = ["in0"], portAnnotations = [], portSyms = [""]} : () -> ()
}

// -----

firrtl.circuit "foo" {
// expected-error @+1 {{block argument types should match signature types}}
"firrtl.module"() ( {
  ^entry(%a: i1):
}) {sym_name = "foo", portTypes = [!firrtl.uint], portDirections = 1 : i1,
    portNames = ["in0"], portAnnotations = [], portSyms = [""]} : () -> ()
}

// -----

firrtl.circuit "Foo" {
firrtl.module @Foo() {
  // expected-error @+1 {{invalid kind of type specified}}
  firrtl.constant 100 : !firrtl.bundle<>
}
}

// -----

firrtl.circuit "Foo" {
firrtl.module @Foo() {
  // expected-error @+1 {{constant too large for result type}}
  firrtl.constant 100 : !firrtl.uint<4>
}
}

// -----

firrtl.circuit "Foo" {
firrtl.module @Foo() {
  // expected-error @+1 {{constant too large for result type}}
  firrtl.constant -100 : !firrtl.sint<4>
}
}

// -----

firrtl.circuit "Foo" {
firrtl.module @Foo() {
  // expected-error @+1 {{special constants can only be 0 or 1}}
  firrtl.specialconstant 2 : !firrtl.clock
}
}

// -----

firrtl.circuit "Foo" {
firrtl.module @Foo() {
  // expected-error @+1 {{special constants can only be 0 or 1}}
  firrtl.specialconstant 2 : !firrtl.reset
}
}

// -----

firrtl.circuit "Foo" {
firrtl.module @Foo() {
  // expected-error @+1 {{special constants can only be 0 or 1}}
  firrtl.specialconstant 2 : !firrtl.asyncreset
}
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %clk: !firrtl.clock, in %reset: !firrtl.uint<2>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // expected-error @+1 {{'firrtl.regreset' op operand #1 must be Reset, but got '!firrtl.uint<2>'}}
    %a = firrtl.regreset %clk, %reset, %zero {name = "a"} : !firrtl.uint<2>, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo() {
  // expected-error @+1 {{'firrtl.mem' op attribute 'writeLatency' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 1}}
    %m = firrtl.mem Undefined {depth = 32 : i64, name = "m", readLatency = 0 : i32, writeLatency = 0 : i32} : !firrtl.bundle<>
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-error @+1 {{'firrtl.extmodule' op attribute 'defname' with value "Bar" conflicts with the name of another module in the circuit}}
  firrtl.extmodule @Foo() attributes { defname = "Bar" }
  // expected-note @+1 {{previous module declared here}}
  firrtl.module @Bar() {}
  // Allow an extmodule to conflict with its own symbol name
  firrtl.extmodule @Baz() attributes { defname = "Baz" }

}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Foo(in a : !firrtl.uint<1>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has 0 ports which is different from a previously defined extmodule with the same 'defname' which has 1 ports}}
  firrtl.extmodule @Bar() attributes { defname = "Foo" }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Foo(in a : !firrtl.uint<1>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has a port with name "b" which does not match the name of the port in the same position of a previously defined extmodule with the same 'defname', expected port to have name "a"}}
  firrtl.extmodule @Foo_(in b : !firrtl.uint<1>) attributes { defname = "Foo" }
}

// -----

firrtl.circuit "Foo" {
  firrtl.extmodule @Foo<width: i32 = 2>(in a : !firrtl.uint<2>) attributes { defname = "Foo" }
  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Bar(in a : !firrtl.uint<1>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has a port with name "a" which has a different type '!firrtl.uint<2>' which does not match the type of the port in the same position of a previously defined extmodule with the same 'defname', expected port to have type '!firrtl.uint<1>'}}
  firrtl.extmodule @Baz(in a : !firrtl.uint<2>) attributes { defname = "Foo" }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Foo(in a : !firrtl.uint<1>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has a port with name "a" which has a different type '!firrtl.sint<1>' which does not match the type of the port in the same position of a previously defined extmodule with the same 'defname', expected port to have type '!firrtl.uint<1>'}}
  firrtl.extmodule @Foo_(in a : !firrtl.sint<1>) attributes { defname = "Foo" }
}

// -----

firrtl.circuit "Foo" {

  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Foo<width: i32 = 2>(in a : !firrtl.uint<2>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has a port with name "a" which has a different type '!firrtl.sint' which does not match the type of the port in the same position of a previously defined extmodule with the same 'defname', expected port to have type '!firrtl.uint'}}
  firrtl.extmodule @Bar(in a : !firrtl.sint<1>) attributes { defname = "Foo" }

}

// -----

firrtl.circuit "Foo" {
  // expected-error @+1 {{has unknown extmodule parameter value 'width' = @Foo}}
  firrtl.extmodule @Foo<width: none = @Foo>(in a : !firrtl.uint<2>) attributes { defname = "Foo" }
}

// -----

firrtl.circuit "Foo" {
  firrtl.extmodule @Foo()
  // expected-error @+1 {{'firrtl.instance' op should be embedded in a 'firrtl.module'}}
  firrtl.instance "" @Foo()
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{containing module declared here}}
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op is a recursive instantiation of its containing module}}
    firrtl.instance "" @Foo()
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{original module declared here}}
  firrtl.module @Callee(in %arg0: !firrtl.uint<1>) { }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op result type for "arg0" must be '!firrtl.uint<1>', but got '!firrtl.uint<2>'}}
    %a = firrtl.instance "" @Callee(in arg0: !firrtl.uint<2>)
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{original module declared here}}
  firrtl.module @Callee(in %arg0: !firrtl.uint<1> ) { }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op has a wrong number of results; expected 1 but got 0}}
    firrtl.instance "" @Callee()
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{original module declared here}}
  firrtl.module @Callee(in %arg0: !firrtl.uint<1>, in %arg1: !firrtl.bundle<valid: uint<1>>) { }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op result type for "arg1" must be '!firrtl.bundle<valid: uint<1>>', but got '!firrtl.bundle<valid: uint<2>>'}}
    %a:2 = firrtl.instance "" @Callee(in arg0: !firrtl.uint<1>, in arg1: !firrtl.bundle<valid: uint<2>>)
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{original module declared here}}
  firrtl.module @Callee(in %arg0: !firrtl.uint<1>, in %arg1: !firrtl.bundle<valid: uint<1>>) { }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op name for port 1 must be "arg1", but got "xxx"}}
    %a:2 = firrtl.instance "" @Callee(in arg0: !firrtl.uint<1>, in xxx: !firrtl.bundle<valid: uint<1>>)
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{original module declared here}}
  firrtl.module @Callee(in %arg0: !firrtl.uint<1>, in %arg1: !firrtl.bundle<valid: uint<1>>) { }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op direction for "arg1" must be "in", but got "out"}}
    %a:2 = firrtl.instance "" @Callee(in arg0: !firrtl.uint<1>, out arg1: !firrtl.bundle<valid: uint<1>>)
  }
}

// -----

firrtl.circuit "X" {

firrtl.module @X(in %a : !firrtl.uint<4>) {
  // expected-error @+1 {{high must be equal or greater than low, but got high = 3, low = 4}}
  %0 = firrtl.bits %a 3 to 4 : (!firrtl.uint<4>) -> !firrtl.uint<2>
}

}

// -----

firrtl.circuit "X" {

firrtl.module @X(in %a : !firrtl.uint<4>) {
  // expected-error @+1 {{high must be smaller than the width of input, but got high = 4, width = 4}}
  %0 = firrtl.bits %a 4 to 3 : (!firrtl.uint<4>) -> !firrtl.uint<2>
}

}

// -----

firrtl.circuit "X" {

firrtl.module @X(in %a : !firrtl.uint<4>) {
  // expected-error @+1 {{'firrtl.bits' op inferred type(s) '!firrtl.uint<3>' are incompatible with return type(s) of operation '!firrtl.uint<2>'}}
  %0 = firrtl.bits %a 3 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<2>
}

}

// -----

firrtl.circuit "BadPort" {
  firrtl.module @BadPort(in %a : !firrtl.uint<1>) {
    // expected-error @+1 {{'firrtl.attach' op operand #0 must be analog type, but got '!firrtl.uint<1>'}}
    firrtl.attach %a, %a : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "BadAdd" {
  firrtl.module @BadAdd(in %a : !firrtl.uint<1>) {
    // expected-error @+1 {{'firrtl.add' op inferred type(s) '!firrtl.uint<2>' are incompatible with return type(s) of operation '!firrtl.uint<1>'}}
    firrtl.add %a, %a : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "StructCast" {
  firrtl.module @StructCast() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
    // expected-error @+1 {{bundle and struct have different number of fields}}
    %b = firrtl.hwStructCast %a : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>) -> (!hw.struct<valid: i1, ready: i1>)
  }
}

// -----

firrtl.circuit "StructCast2" {
  firrtl.module @StructCast2() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<1>>
    // expected-error @+1 {{field names don't match 'valid', 'yovalid'}}
    %b = firrtl.hwStructCast %a : (!firrtl.bundle<valid: uint<1>>) -> (!hw.struct<yovalid: i1>)
  }
}

// -----

firrtl.circuit "StructCast3" {
  firrtl.module @StructCast3() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<1>>
    // expected-error @+1 {{size of field 'valid' don't match 1, 2}}
    %b = firrtl.hwStructCast %a : (!firrtl.bundle<valid: uint<1>>) -> (!hw.struct<valid: i2>)
  }
}

// -----

firrtl.circuit "OutOfOrder" {
  firrtl.module @OutOfOrder(in %a: !firrtl.uint<32>) {
    // expected-error @+1 {{operand #0 does not dominate this use}}
    %0 = firrtl.add %1, %1 : (!firrtl.uint<33>, !firrtl.uint<33>) -> !firrtl.uint<34>
    // expected-note @+1 {{operand defined here}}
    %1 = firrtl.add %a, %a : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
  }
}

// -----

firrtl.circuit "CombMemInvalidReturnType" {
  firrtl.module @CombMemInvalidReturnType() {
    // expected-error @+1 {{'chirrtl.combmem' op result #0 must be a behavioral memory, but got '!firrtl.uint<1>'}}
    %mem = chirrtl.combmem : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "CombMemNonPassiveReturnType" {
  firrtl.module @CombMemNonPassiveReturnType() {
    // expected-error @+1 {{behavioral memory element type must be passive}}
    %mem = chirrtl.combmem : !chirrtl.cmemory<bundle<a flip : uint<1>>, 1>
  }
}

// -----

firrtl.circuit "SeqMemInvalidReturnType" {
  firrtl.module @SeqMemInvalidReturnType() {
    // expected-error @+1 {{'chirrtl.seqmem' op result #0 must be a behavioral memory, but got '!firrtl.uint<1>'}}
    %mem = chirrtl.seqmem Undefined : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "SeqMemNonPassiveReturnType" {
  firrtl.module @SeqMemNonPassiveReturnType() {
    // expected-error @+1 {{behavioral memory element type must be passive}}
    %mem = chirrtl.seqmem Undefined : !chirrtl.cmemory<bundle<a flip : uint<1>>, 1>
  }
}

// -----

firrtl.circuit "MemoryPortInvalidReturnType" {
  firrtl.module @MemoryPortInvalidReturnType(in %sel : !firrtl.uint<8>, in %clock : !firrtl.clock) {
    %mem = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 8>
    // expected-error @+1 {{'chirrtl.memoryport' op port should be used by a chirrtl.memoryport.access}}
    %memoryport_data, %memoryport_port = chirrtl.memoryport Infer %mem {name = "memoryport"} : (!chirrtl.cmemory<uint<8>, 8>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
  }
}

// -----

firrtl.circuit "MemoryPortInvalidReturnType" {
  firrtl.module @MemoryPortInvalidReturnType(in %sel : !firrtl.uint<8>, in %clock : !firrtl.clock) {
    %mem = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 8>
    // expected-error @+1 {{'chirrtl.memoryport' op inferred type(s) '!firrtl.uint<8>', '!chirrtl.cmemoryport' are incompatible with return type(s) of operation '!firrtl.uint<9>', '!chirrtl.cmemoryport'}}
    %memoryport_data, %memoryport_port = chirrtl.memoryport Infer %mem {name = "memoryport"} : (!chirrtl.cmemory<uint<8>, 8>) -> (!firrtl.uint<9>, !chirrtl.cmemoryport)
  }
}

// -----

firrtl.circuit "MemoryInvalidmask" {
  firrtl.module @MemoryInvalidmask() {
    // expected-error @+1 {{'firrtl.mem' op the mask width cannot be greater than data width}}
    %memory_rw = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["rw"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<9>>
  }
}
// -----

firrtl.circuit "MemoryNegativeReadLatency" {
  firrtl.module @MemoryNegativeReadLatency() {
    // expected-error @+1 {{'firrtl.mem' op attribute 'readLatency' failed to satisfy constraint}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "rw", "w"], readLatency = -1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
}

// -----

firrtl.circuit "MemoryZeroWriteLatency" {
  firrtl.module @MemoryZeroWriteLatency() {
    // expected-error @+1 {{'firrtl.mem' op attribute 'writeLatency' failed to satisfy constraint}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "rw", "w"], readLatency = 0 : i32, writeLatency = 0 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
}

// -----

firrtl.circuit "MemoryZeroDepth" {
  firrtl.module @MemoryZeroDepth() {
    // expected-error @+1 {{'firrtl.mem' op attribute 'depth' failed to satisfy constraint}}
    %memory_r = firrtl.mem Undefined {depth = 0 : i64, name = "memory", portNames = ["r", "rw", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
}

// -----

firrtl.circuit "MemoryBadPortType" {
  firrtl.module @MemoryBadPortType() {
    // expected-error @+1 {{'firrtl.mem' op has an invalid type on port "r" (expected '!firrtl.bundle<...>'}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "MemoryPortNamesCollide" {
  firrtl.module @MemoryPortNamesCollide() {
    // expected-error @+1 {{'firrtl.mem' op has non-unique port name "r"}}
    %memory_r, %memory_r_0 = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
}

// -----

firrtl.circuit "MemoryUnexpectedNumberOfFields" {
  firrtl.module @MemoryUnexpectedNumberOfFields() {
    // expected-error @+1 {{'firrtl.mem' op has an invalid number of fields on port "r" (expected 4 for read, 5 for write, or 7 for read/write)}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<a: uint<1>>
  }
}

// -----

firrtl.circuit "MemoryMissingDataField" {
  firrtl.module @MemoryMissingDataField() {
    // expected-error @+1 {{'firrtl.mem' op has no data field on port "r" (expected to see "data" for a read or write port or "rdata" for a read/write port)}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>>
  }
}

// -----

firrtl.circuit "MemoryMissingDataField2" {
  firrtl.module @MemoryMissingDataField2() {
    // expected-error @+1 {{'firrtl.mem' op has no data field on port "rw" (expected to see "data" for a read or write port or "rdata" for a read/write port)}}
    %memory_rw = firrtl.mem Undefined {depth = 16 : i64, name = "memory2", portNames = ["rw"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, writedata: uint<8>, wmask: uint<1>>
  }
}

// -----

firrtl.circuit "MemoryDataNotPassive" {
  firrtl.module @MemoryDataNotPassive() {
    // expected-error @+1 {{'firrtl.mem' op has non-passive data type on port "r" (memory types must be passive)}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a flip: uint<8>, b: uint<8>>>
  }
}

// -----

firrtl.circuit "MemoryDataContainsAnalog" {
  firrtl.module @MemoryDataContainsAnalog() {
    // expected-error @+1 {{'firrtl.mem' op has a data type that contains an analog type on port "r" (memory types cannot contain analog types)}}
    %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: analog<8>>>
  }
}

// -----

firrtl.circuit "MemoryPortInvalidReadKind" {
  firrtl.module @MemoryPortInvalidReadKind() {
    // expected-error @+1 {{'firrtl.mem' op has an invalid type for port "r" of determined kind "read" (expected '!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>', but got '!firrtl.bundle<BAD: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>')}}
    %memory_r= firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<BAD: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
}

// -----

firrtl.circuit "MemoryPortInvalidWriteKind" {
  firrtl.module @MemoryPortInvalidWriteKind() {
    // expected-error @+1 {{'firrtl.mem' op has an invalid type for port "w" of determined kind "write" (expected '!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>', but got '!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, BAD: uint<1>>')}}
    %memory_r= firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, BAD: uint<1>>
  }
}

// -----

firrtl.circuit "MemoryPortInvalidReadWriteKind" {
  firrtl.module @MemoryPortInvalidReadWriteKind() {
    // expected-error @+1 {{'firrtl.mem' op has an invalid type for port "rw" of determined kind "readwrite" (expected '!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>', but got '!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, BAD: uint<1>>')}}
    %memory_r= firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["rw"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, BAD: uint<1>>
  }
}

// -----

firrtl.circuit "MemoryPortsWithDifferentTypes" {
  firrtl.module @MemoryPortsWithDifferentTypes() {
    // expected-error @+1 {{'firrtl.mem' op port "r1" has a different type than port "r0" (expected '!firrtl.uint<8>', but got '!firrtl.sint<8>')}}
    %memory_r0, %memory_r1 = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r0", "r1"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
  }
}

// -----

firrtl.circuit "SubfieldOpFieldError" {
  firrtl.module @SubfieldOpFieldError() {
    %w = firrtl.wire  : !firrtl.bundle<a: uint<2>, b: uint<2>>
    // expected-error @+1 {{}}
    %w_a = firrtl.subfield %w(2) : (!firrtl.bundle<a : uint<2>, b : uint<2>>) -> !firrtl.uint<2>
  }
}

// -----

firrtl.circuit "BitCast1" {
  firrtl.module @BitCast1() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint>
    // expected-error @+1 {{bitwidth cannot be determined for input operand type}}
    %b = firrtl.bitcast %a : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint>) -> (!firrtl.uint<6>)
  }
}

// -----

firrtl.circuit "BitCast2" {
  firrtl.module @BitCast2() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>
    // expected-error @+1 {{the bitwidth of input (3) and result (6) don't match}}
    %b = firrtl.bitcast %a : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>) -> (!firrtl.uint<6>)
  }
}

// -----

firrtl.circuit "BitCast4" {
  firrtl.module @BitCast4() {
    %a = firrtl.wire : !firrtl.bundle<valid flip: uint<1>, ready: uint<1>, data: uint<1>>
    // expected-error @+1 {{bitwidth cannot be determined for input operand type}}
    %b = firrtl.bitcast %a : (!firrtl.bundle<valid flip: uint<1>, ready: uint<1>, data: uint<1>>) -> (!firrtl.uint<6>)
  }
}

// -----


firrtl.circuit "LowerToBind" {
 // expected-error @+1 {{the instance path cannot be empty/single element}}
firrtl.nla @NLA1 []
firrtl.nla @NLA2 [#hw.innerNameRef<@LowerToBind::@s1>]
firrtl.module @InstanceLowerToBind() {}
firrtl.module @LowerToBind() {
  firrtl.instance foo sym @s1 {lowerToBind = true, annotations = [{circt.nonlocal = @NLA2, class = "circt.test", nl = "nl"}]} @InstanceLowerToBind() 
}
}

// -----

firrtl.circuit "NLATop" {

 // expected-error @+1 {{the instance path can only contain inner sym reference, only the leaf can refer to a module symbol}}
  firrtl.nla @nla [#hw.innerNameRef<@NLATop::@test>, @Aardvark, @Zebra]
  firrtl.nla @nla_1 [#hw.innerNameRef<@NLATop::@test>,#hw.innerNameRef<@Aardvark::@test_1>, @Zebra]
  firrtl.module @NLATop() {
    firrtl.instance test  sym @test {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_1, class = "circt.nonlocal"} ]}@Aardvark()
    firrtl.instance test2 @Zebra()
  }

  firrtl.module @Aardvark() {
    firrtl.instance test sym @test {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]}@Zebra()
    firrtl.instance test1 sym @test_1 {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]}@Zebra()
  }

  firrtl.module @Zebra() {
  }
}

// -----

firrtl.circuit "NLATop1" {
  // expected-error @+1 {{instance path is incorrect. Expected module: "Aardvark" instead found: "Zebra"}}
  firrtl.nla @nla [#hw.innerNameRef<@NLATop1::@test>, #hw.innerNameRef<@Zebra::@test>,#hw.innerNameRef<@Aardvark::@test>]
  firrtl.nla @nla_1 [#hw.innerNameRef<@NLATop1::@test>,#hw.innerNameRef<@Aardvark::@test_1>, @Zebra]
  firrtl.module @NLATop1() {
    firrtl.instance test  sym @test {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_1, class = "circt.nonlocal"} ]}@Aardvark()
    firrtl.instance test2 @Zebra()
  }

  firrtl.module @Aardvark() {
    firrtl.instance test sym @test {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]}@Zebra()
    firrtl.instance test1 sym @test_1 {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]}@Zebra()
  }

  firrtl.module @Zebra() {
    firrtl.instance test sym @test {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]}@Ext()
    firrtl.instance test1 sym @test_1 {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]}@Ext()
  }

  firrtl.module @Ext() {
  }
}

// -----

// This should not error out. Note that there is no symbol on the %bundle. This handles a special case, when the nonlocal is applied to a subfield.
firrtl.circuit "fallBackName" {

  firrtl.nla @nla [#hw.innerNameRef<@fallBackName::@test>, #hw.innerNameRef<@Aardvark::@test>, #hw.innerNameRef<@Zebra::@bundle>]
  firrtl.nla @nla_1 [#hw.innerNameRef<@fallBackName::@test>,#hw.innerNameRef<@Aardvark::@test_1>, @Zebra]
  firrtl.module @fallBackName() {
    firrtl.instance test  sym @test {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_1, class = "circt.nonlocal"} ]}@Aardvark()
    firrtl.instance test2 @Zebra()
  }

  firrtl.module @Aardvark() {
    firrtl.instance test sym @test {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]}@Zebra()
    firrtl.instance test1 sym @test_1 {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]}@Zebra()
  }

  firrtl.module @Zebra(){
    %bundle = firrtl.wire  sym @bundle {annotations = [#firrtl<"subAnno<fieldID = 2, {circt.nonlocal = @nla, class =\"test\" }>">]}: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
  }
}

// -----

firrtl.circuit "Foo"   {
  // expected-error @+1 {{operation with symbol: #hw.innerNameRef<@Bar::@b> was not found}}
  firrtl.nla @nla_1 [#hw.innerNameRef<@Foo::@bar>, #hw.innerNameRef<@Bar::@b>]
  firrtl.module @Bar(in %a: !firrtl.uint<1>, out %b: !firrtl.bundle<baz: uint<1>, qux: uint<1>> [#firrtl.subAnno<fieldID = 2, {circt.nonlocal = @nla_1, three}>], out %c: !firrtl.uint<1>) {
  }
  firrtl.module @Foo() {
    %bar_a, %bar_b, %bar_c = firrtl.instance bar sym @bar  {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]} @Bar(in a: !firrtl.uint<1> [{one}], out b: !firrtl.bundle<baz: uint<1>, qux: uint<1>> [#firrtl.subAnno<fieldID = 1, {two}>], out c: !firrtl.uint<1> [{four}])
  }
}

// -----

firrtl.circuit "Top"   {
 // Legal nla would be:
//firrtl.nla @nla [#hw.innerNameRef<@Top::@mid>, #hw.innerNameRef<@Mid::@leaf>, #hw.innerNameRef<@Leaf::@w>]
  // expected-error @+1 {{instance path is incorrect. Expected module: "Middle" instead found: "Leaf"}}
  firrtl.nla @nla [#hw.innerNameRef<@Top::@mid>, #hw.innerNameRef<@Leaf::@w>]
  firrtl.module @Leaf() {
    %w = firrtl.wire sym @w  {annotations = [{circt.nonlocal = @nla, class = "fake1"}]} : !firrtl.uint<3>
  }
  firrtl.module @Middle() {
    firrtl.instance leaf sym @leaf  {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]} @Leaf()
  }
  firrtl.module @Top() {
    firrtl.instance mid sym @mid  {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]} @Middle()
  }
}

