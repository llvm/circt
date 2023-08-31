// RUN: circt-opt -firrtl-lower-classes %s -verify-diagnostics --split-input-file

firrtl.circuit "Component" {
  firrtl.module @Component() {}
  // expected-error @+1{{failed to legalize operation 'om.class' that was explicitly marked illegal}}
  firrtl.class @Map(in %s1: !firrtl.map<list<string>, string>) {}
}

// -----

firrtl.circuit "PathNoID" {
  firrtl.module @PathNoID() {
    // expected-error @below {{circt.tracker annotation missing id field}}
    %wire = firrtl.wire {annotations = [{class = "circt.tracker"}]} : !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "PathIllegalHierpath" {
  firrtl.module @Path() {}
  firrtl.module @PathIllegalHierpath() {
    // expected-error @below {{annotation does not point at a HierPathOp}}
    %wire = firrtl.wire {annotations = [{circt.nonlocal = @Path, class = "circt.tracker", id = distinct[0]<>}]} : !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "PathDuplicateID" {
  firrtl.module @PathDuplicateID() {
    // expected-error @below {{duplicate identifier found}}
    %a = firrtl.wire {annotations = [{class = "circt.tracker", id = distinct[0]<>}]} : !firrtl.uint<8>
    // expected-note @below {{other identifier here}}
    %b = firrtl.wire {annotations = [{class = "circt.tracker", id = distinct[0]<>}]} : !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "PathDontTouchDeleted" {
  firrtl.module @PathDontTouchDeleted() {}
  firrtl.class @Class() {
    // expected-error @+2 {{DontTouch target was deleted}}
    // expected-error @+1 {{failed to legalize operation 'firrtl.path' that was explicitly marked illegal}}
    %0 = firrtl.path dont_touch distinct[0]<>
  }
}
