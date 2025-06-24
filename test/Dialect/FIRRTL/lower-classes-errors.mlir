// RUN: circt-opt -firrtl-lower-classes %s -verify-diagnostics --split-input-file

firrtl.circuit "UnassignedInputPort" {
  firrtl.class @Class(in %input: !firrtl.string) {}

  firrtl.module @UnassignedInputPort() {
    // expected-error @below {{uninitialized input port "input"}}
    %obj = firrtl.object @Class(in input: !firrtl.string)
  }
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
    // Duplicate ID is only an error if something actually refers to that ID. Dedup can create dead, duplicate IDs.
    %path = firrtl.path reference distinct[0]<>

    // expected-error @below {{path identifier already found, paths must resolve to a unique target}}
    %a = firrtl.wire {annotations = [{class = "circt.tracker", id = distinct[0]<>}]} : !firrtl.uint<8>
    // expected-note @below {{other path identifier here}}
    %b = firrtl.wire {annotations = [{class = "circt.tracker", id = distinct[0]<>}]} : !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "PathDontTouchDeleted" {
  firrtl.module @PathDontTouchDeleted() {
    %class = firrtl.object @Class()
  }
  firrtl.class @Class() {
    // expected-error @+2 {{DontTouch target was deleted}}
    // expected-error @+1 {{failed to legalize operation 'firrtl.path' that was explicitly marked illegal}}
    %0 = firrtl.path dont_touch distinct[0]<>
  }
}

// -----

firrtl.circuit "PathInstanceDeleted" {
  firrtl.module @PathInstanceDeleted() {
    %class = firrtl.object @Class()
  }
  firrtl.class @Class() {
    // expected-error @+2 {{Instance target was deleted}}
    // expected-error @+1 {{failed to legalize operation 'firrtl.path' that was explicitly marked illegal}}
    %0 = firrtl.path instance distinct[0]<>
  }
}

// -----

firrtl.circuit "NotInstance" {
  firrtl.module @NotInstance() {
    // expected-note @below {{target not instance or module}}
    firrtl.wire {annotations = [{class = "circt.tracker", id = distinct[0]<>}]} : !firrtl.uint<8>
    %class = firrtl.object @Class()
  }
  firrtl.class @Class() {
    // expected-error @below {{invalid target for instance path}}
    // expected-error @below {{failed to legalize operation 'firrtl.path' that was explicitly marked illegal}}
    %0 = firrtl.path instance distinct[0]<>
  }
}
