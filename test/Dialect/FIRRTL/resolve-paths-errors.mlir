// RUN: circt-opt %s -firrtl-resolve-paths -split-input-file -verify-diagnostics

firrtl.circuit "BadTargetKind" {
firrtl.module @BadTargetKind() {
    // expected-error @below {{unknown or missing OM reference type in target string: "ASDFA:"}}
    %0 = firrtl.unresolved_path "ASDFA:"
}
}

// -----

firrtl.circuit "BadPathString" {
firrtl.module @BadPathString() {
    // expected-error @below {{OMDeleted references can not have targets}}
    %0 = firrtl.unresolved_path "OMDeleted:"
}
}

// -----

firrtl.circuit "BadPathString" {
firrtl.module @BadPathString() {
    // expected-error @below {{expected ':' in target string}}
    %0 = firrtl.unresolved_path "OMReferenceTarget"
}
}

// -----

firrtl.circuit "BadPathString" {
firrtl.module @BadPathString() {
    // expected-error @below {{cannot tokenize annotation path ""}}
    %0 = firrtl.unresolved_path "OMReferenceTarget:"
}
}

// -----

firrtl.circuit "NonHardwareTarget" {
firrtl.module @NonHardwareTarget(in %a : !firrtl.string) {
    // expected-error @below {{unable to target non-hardware type '!firrtl.string'}}
    %0 = firrtl.unresolved_path "OMReferenceTarget:~NonHardwareTarget|NonHardwareTarget>a"
}
}

// -----

firrtl.circuit "BundleTarget" {
firrtl.module @BundleTarget(in %a : !firrtl.bundle<>) {
    // expected-error @below {{unable to target aggregate type '!firrtl.bundle<>'}}
    %0 = firrtl.unresolved_path "OMReferenceTarget:~BundleTarget|BundleTarget>a"
}
}

// -----

firrtl.circuit "VectorTarget" {
firrtl.module @VectorTarget(in %a : !firrtl.vector<uint<1>, 1>) {
    // expected-error @below {{unable to target aggregate type '!firrtl.vector<uint<1>, 1>'}}
    %0 = firrtl.unresolved_path "OMReferenceTarget:~VectorTarget|VectorTarget>a"
}
}

// -----

firrtl.circuit "AmbiguousPath" {
firrtl.module @AmbiguousPath() {
    // expected-error @below {{unable to uniquely resolve target due to multiple instantiation}}
    %0 = firrtl.unresolved_path "OMReferenceTarget:~AmbiguousPath|Child"
    // expected-note @below {{instance here}}
    firrtl.instance child0 @Child()
    // expected-note @below {{instance here}}
    firrtl.instance child1 @Child()
}
firrtl.module @Child() {}
}

// -----

firrtl.circuit "NoOwningModule" {
firrtl.module @NoOwningModule() {
    %wire = firrtl.wire : !firrtl.uint<8>
}

firrtl.class @Test(){
  %om = firrtl.object @Test()
  // expected-error @below {{path does not have a single owning module}}
  %0 = firrtl.unresolved_path "OMReferenceTarget:~NoOwningModule|NoOwningModule>wire"
}
}

// -----

firrtl.circuit "UpwardPath" {
firrtl.module @UpwardPath() {
  %wire = firrtl.wire : !firrtl.uint<8>
  firrtl.instance child @Child()
}

firrtl.module @Child() {
  %om = firrtl.object @OM()

}

firrtl.class @OM(){
  // expected-error @below {{unable to resolve path relative to owning module "Child"}}
  %0 = firrtl.unresolved_path "OMReferenceTarget:~UpwardPath|UpwardPath>wire"
}
}

