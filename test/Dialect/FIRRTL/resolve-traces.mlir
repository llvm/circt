// RUN: circt-opt %s -firrtl-resolve-traces | FileCheck %s

firrtl.circuit "Foo" {
  firrtl.module @Foo() attributes {annotations = [
    {
      class = "chisel3.experimental.Trace$TraceAnnotation",
      chiselTarget = "~Foo|Original"
    }
  ]} {}
}

// Test that a local module annotation is resolved.
//
// CHECK:      class{{.*}}:{{.*}}chisel3.experimental.Trace$TraceAnnotation
// CHECK-SAME: target{{.*}}:{{.*}}~Foo|Foo
// CHECK-SAME: chiselTarget{{.*}}:{{.*}}~Foo|Original

// Test that the output file is set to include the circuit name.
//
// CHECK-SAME: #hw.output_file<{{.*}}Foo.anno.json

// -----

firrtl.circuit "Foo" {
  hw.hierpath @path [@Foo::@bar, @Bar]
  firrtl.module @Bar() attributes {annotations = [
    {
      class = "chisel3.experimental.Trace$TraceAnnotation",
      chiselTarget = "~Foo|Original",
      circt.nonlocal = @path
    }
  ]} {}
  firrtl.module @Foo() {
    firrtl.instance bar sym @bar @Bar()
  }
}

// Test that a non-local module annotation is resolved.
//
// CHECK: ~Foo|Foo/bar:Bar{{.*}}~Foo|Original

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(
    in %a: !firrtl.uint<1> [
      {
        class = "chisel3.experimental.Trace$TraceAnnotation",
        chiselTarget = "~Foo|Foo>original"
      }
    ]
  ) {}
}

// Test that a local port annotation is resolved.
//
// CHECK: ~Foo|Foo>a{{.*}}~Foo|Foo>original

// Test that the port receives a symbol.
//
// CHECK: in %a: !firrtl.uint<1> sym

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo() {
    %a = firrtl.wire {annotations = [
      {
        class = "chisel3.experimental.Trace$TraceAnnotation",
        chiselTarget = "~Foo|Foo>original"
      }
    ]} : !firrtl.uint<1>
  }
}

// Test that a local wire annotation is resolved.
//
// CHECK: ~Foo|Foo>a{{.*}}~Foo|Foo>original

// Test that the wire receives a symbol.
//
// CHECK: %a = firrtl.wire sym

// -----

firrtl.circuit "Forceable" {
  firrtl.module @Forceable() {
    %w, %w_ref = firrtl.wire forceable {annotations = [
      {
        class = "chisel3.experimental.Trace$TraceAnnotation",
        chiselTarget = "~Forceable|Forceable>forced"
      }
    ]} : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
  }
}

// Test that a local wire annotation is resolved.
//
// CHECK: ~Forceable|Forceable>w{{.*}}~Forceable|Forceable>forced

// Test that the wire receives a symbol.
//
// CHECK-LABEL: firrtl.circuit "Forceable"
// CHECK: %w, %w_ref = firrtl.wire sym

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo() {
    %a = firrtl.wire {annotations = [
      {
        class = "chisel3.experimental.Trace$TraceAnnotation",
        chiselTarget = "~Foo|Foo>0",
        circt.fieldID = 0
      },
      {
        class = "chisel3.experimental.Trace$TraceAnnotation",
        chiselTarget = "~Foo|Foo>4",
        circt.fieldID = 4
      },
      {
        class = "chisel3.experimental.Trace$TraceAnnotation",
        chiselTarget = "~Foo|Foo>5",
        circt.fieldID = 6
      }
    ]} : !firrtl.vector<bundle<a: uint<1>, b: uint<1>>, 2>
  }
}

// Test that a local wire annotation on an aggregate is resolved for different
// values of field ID.
//
// CHECK:      ~Foo|Foo>a{{.*}}~Foo|Foo>0
// CHECK-SAME: ~Foo|Foo>a[1]{{.*}}~Foo|Foo>4
// CHECK-SAME: ~Foo|Foo>a[1].b{{.*}}~Foo|Foo>5
