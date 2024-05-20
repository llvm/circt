//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-materialize-debug-info)))' --verify-diagnostics %s 

firrtl.circuit "ExpectedError" {
// CHECK-LABEL: @ExpectedError
firrtl.module @ExpectedError() {
  // expected-error @below {{Missing expected typeName in tywaves annotation.}}
  %a = firrtl.wire {annotations = [{class = "chisel3.tywaves.TywavesAnnotation", typeNam = "Wire[SInt<17>]"}]} : !firrtl.uint<17> 
  
  // expected-error @below {{Parameters are expressed in the wrong format, expected array. Found: {name = "size", typeName = "int", value = "17"}}}
  %b = firrtl.wire {annotations = [{class = "chisel3.tywaves.TywavesAnnotation", typeName = "Wire[SInt<17>]", params = {name="size", typeName="int", value="17"}}]} : !firrtl.uint<17>
  // expected-error @below {{Parameters are missing one of the following fields: name, typeName. Found: ["a", 10]}}
  %b2 = firrtl.wire {annotations = [{class = "chisel3.tywaves.TywavesAnnotation", typeName = "Wire[SInt<17>]", params = ["a", 10]}]} : !firrtl.uint<17>
  // expected-error @below {{Parameters are missing one of the following fields: name, typeName. Found: [{nam = "size", typeName = "int", value = "17"}]}}
  %c = firrtl.wire {annotations = [{class = "chisel3.tywaves.TywavesAnnotation", typeName = "Wire[SInt<17>]", params = [{nam="size", typeName="int", value="17"}]}]} : !firrtl.uint<17>
  // expected-error @below {{Parameters are missing one of the following fields: name, typeName. Found: [{name = "size", typ = "int", value = "17"}]}}
  %d = firrtl.wire {annotations = [{class = "chisel3.tywaves.TywavesAnnotation", typeName = "Wire[SInt<17>]", params = [{name="size", typ="int", value="17"}]}]} : !firrtl.uint<17>
  // no error
  %e = firrtl.wire {annotations = [{class = "chisel3.tywaves.TywavesAnnotation", typeName = "Wire[SInt<17>]", params = [{name="size", typeName="int", value="17"}]}]} : !firrtl.uint<17>
  // no error: value is optional
  %f = firrtl.wire {annotations = [{class = "chisel3.tywaves.TywavesAnnotation", typeName = "Wire[SInt<17>]", params = [{name="size", typeName="int"}]}]} : !firrtl.uint<17>
  
  // expected-error @below {{Expected a single tywaves annotation. Found: 3}}
  %g = firrtl.wire {annotations = [{class = "chisel3.tywaves.TywavesAnnotation", typeName = "Wire[SInt<17>]"},
                                   {class = "chisel3.tywaves.TywavesAnnotation", typeName = "Wire[UInt<17>]"},
                                   {class = "chisel3.tywaves.TywavesAnnotation", typeName = "Wire[SInt<17>]", params = [{name="size", typeName="int", value="17"}]}]} : !firrtl.uint<17>
}
}