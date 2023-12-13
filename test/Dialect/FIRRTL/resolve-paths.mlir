// RUN: circt-opt %s -firrtl-resolve-paths -split-input-file | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Deleted"
firrtl.circuit "Deleted" {
firrtl.module @Deleted() {
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    %0 = firrtl.unresolved_path "OMDeleted"
}
}

// -----

// CHECK-LABEL: firrtl.circuit "Deleted"
firrtl.circuit "Deleted" {
firrtl.module @Deleted(out %path : !firrtl.path) {
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    // CHECK: propassign %path, %0 : !firrtl.path
    %0 = firrtl.unresolved_path "OMDeleted"
    firrtl.propassign %path, %0 : !firrtl.path
}
}

// -----

// CHECK-LABEL: firrtl.circuit "TargetTopModule"
firrtl.circuit "TargetTopModule" {
// CHECK: firrtl.module @TargetTopModule() attributes {annotations = [{class = "circt.tracker", id = distinct[0]<>}]}
firrtl.module @TargetTopModule() {
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    %0 = firrtl.unresolved_path "OMReferenceTarget:~TargetTopModule|TargetTopModule"
}
}

// -----

// CHECK-LABEL: firrtl.circuit "TargetChildModule"
firrtl.circuit "TargetChildModule" {
firrtl.module @TargetChildModule() {
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    %0 = firrtl.unresolved_path "OMReferenceTarget:~TargetChildModule|Child"
    firrtl.instance child @Child()
}
// CHECK: firrtl.module @Child() attributes {annotations = [{class = "circt.tracker", id = distinct[0]<>}]}
firrtl.module @Child() {}
}

// -----

// CHECK-LABEL: firrtl.circuit "TargetWire"
firrtl.circuit "TargetWire" {
firrtl.module @TargetWire() {
    // CHECK: %wire = firrtl.wire {annotations = [{class = "circt.tracker", id = distinct[0]<>}]} : !firrtl.uint<8>
    %wire = firrtl.wire : !firrtl.uint<8>
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    %0 = firrtl.unresolved_path "OMReferenceTarget:~TargetWire|TargetWire>wire"
}
}

// -----

// CHECK-LABEL: firrtl.circuit "TargetAggregate"
firrtl.circuit "TargetAggregate" {
firrtl.module @TargetAggregate() {
    // CHECK: %wire = firrtl.wire {annotations = [{circt.fieldID = 1 : i64, class = "circt.tracker", id = distinct[0]<>}]} : !firrtl.bundle<a: uint<8>>
    %wire = firrtl.wire : !firrtl.bundle<a: uint<8>>
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    %0 = firrtl.unresolved_path "OMReferenceTarget:~TargetAggregate|TargetAggregate>wire.a"
}
}

// -----

// CHECK-LABEL: firrtl.circuit "NonLocalPath"
firrtl.circuit "NonLocalPath" {
// CHECK: hw.hierpath private @nla [@NonLocalPath::@sym, @Child]
// CHECK: firrtl.module @NonLocalPath()
firrtl.module @NonLocalPath() {
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    // CHECK: firrtl.instance child0 sym @sym @Child()
    // CHECK: firrtl.instance child1 @Child()
    %0 = firrtl.unresolved_path "OMReferenceTarget:~NonLocalPath|NonLocalPath/child0:Child>wire"
    firrtl.instance child0 @Child()
    firrtl.instance child1 @Child()
}
// CHECK: firrtl.module @Child()
firrtl.module @Child() {
    // CHECK: %wire = firrtl.wire {annotations = [{circt.nonlocal = @nla, class = "circt.tracker", id = distinct[0]<>}]} : !firrtl.uint<8>
    %wire = firrtl.wire : !firrtl.uint<8>
}
}

// -----

// CHECK-LABEL: firrtl.circuit "PathMinimization"
firrtl.circuit "PathMinimization" {
// CHECK: firrtl.module @PathMinimization()
firrtl.module @PathMinimization() {
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    // CHECK: firrtl.instance child @Child()
    %0 = firrtl.unresolved_path "OMReferenceTarget:~PathMinimization|PathMinimization/child:Child>wire"
    firrtl.instance child @Child()
}
// CHECK: firrtl.module @Child()
firrtl.module @Child() {
    // CHECK: %wire = firrtl.wire {annotations = [{class = "circt.tracker", id = distinct[0]<>}]} : !firrtl.uint<8>
    %wire = firrtl.wire : !firrtl.uint<8>
}
}

// -----

// CHECK-LABEL: firrtl.circuit "TargetInstance"
firrtl.circuit "TargetInstance" {
// CHECK: firrtl.module @TargetInstance() {
firrtl.module @TargetInstance() {
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    // CHECK: %1 = firrtl.path instance distinct[1]<>
    // CHECK: %2 = firrtl.path member_instance distinct[2]<>
    // CHECK: firrtl.instance child @Child()
    %0 = firrtl.unresolved_path "OMReferenceTarget:~TargetInstance|TargetInstance/child:Child"
    %1 = firrtl.unresolved_path "OMInstanceTarget:~TargetInstance|TargetInstance/child:Child"
    %2 = firrtl.unresolved_path "OMMemberInstanceTarget:~TargetInstance|TargetInstance/child:Child"
    firrtl.instance child @Child()
}
// CHECK: firrtl.module @Child() attributes {annotations = [
// CHECK-SAME: {class = "circt.tracker", id = distinct[0]<>},
// CHECK-SAME: {class = "circt.tracker", id = distinct[1]<>},
// CHECK-SAME: {class = "circt.tracker", id = distinct[2]<>}
// CHECK-SAME: ]}
firrtl.module @Child() { }
}

// -----

// CHECK-LABEL: firrtl.circuit "TargetInstancePort"
firrtl.circuit "TargetInstancePort" {
// CHECK: firrtl.module @TargetInstancePort() {
firrtl.module @TargetInstancePort() {
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    // CHECK: firrtl.instance child @Child(in in: !firrtl.uint<8>)
    %0 = firrtl.unresolved_path "OMReferenceTarget:~TargetInstancePort|TargetInstancePort/child:Child>in"
    firrtl.instance child @Child(in in : !firrtl.uint<8>)
}
// CHECK: firrtl.module @Child(in %in: !firrtl.uint<8> [{class = "circt.tracker", id = distinct[0]<>}])
firrtl.module @Child(in %in : !firrtl.uint<8>) { }
}

// -----

// CHECK-LABEL: firrtl.circuit "TargetNonlocalInstance"
firrtl.circuit "TargetNonlocalInstance" {
// CHECK: hw.hierpath private @nla [@TargetNonlocalInstance::@sym, @Child]
// CHECK: firrtl.module @TargetNonlocalInstance() {
firrtl.module @TargetNonlocalInstance() {
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    // CHECK: firrtl.instance child0 sym @sym @Child()
    // CHECK: firrtl.instance child1 @Child()
    %0 = firrtl.unresolved_path "OMReferenceTarget:~TargetNonlocalInstance|TargetNonlocalInstance/child0:Child"
    firrtl.instance child0 @Child()
    firrtl.instance child1 @Child()
}
// CHECK: firrtl.module @Child() attributes {annotations = [{circt.nonlocal = @nla, class = "circt.tracker", id = distinct[0]<>}]}
firrtl.module @Child() { }
}

// -----

// CHECK-LABEL: firrtl.circuit "ReuseHierpathOp"
firrtl.circuit "ReuseHierpathOp" {
// CHECK: hw.hierpath private @nla [@ReuseHierpathOp::@sym, @Child]
// CHECK: firrtl.module @ReuseHierpathOp() {
firrtl.module @ReuseHierpathOp() {
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    // CHECK: %1 = firrtl.path reference distinct[1]<>
    // CHECK: firrtl.instance child0 sym @sym @Child()
    // CHECK: firrtl.instance child1 @Child()
    %0 = firrtl.unresolved_path "OMReferenceTarget:~ReuseHierpathOp|ReuseHierpathOp/child0:Child"
    %1 = firrtl.unresolved_path "OMReferenceTarget:~ReuseHierpathOp|ReuseHierpathOp/child0:Child"
    firrtl.instance child0 @Child()
    firrtl.instance child1 @Child()
}
// CHECK: firrtl.module @Child()
// CHECK-SAME: {circt.nonlocal = @nla, class = "circt.tracker", id = distinct[0]<>}
// CHECK-SAME: {circt.nonlocal = @nla, class = "circt.tracker", id = distinct[1]<>}
firrtl.module @Child() { }
}

// -----

// CHECK-LABEL: firrtl.circuit "TargetWire"
firrtl.circuit "TargetWire" {
firrtl.module @TargetWire() {
    %om = firrtl.object @OM()
    // CHECK: %wire = firrtl.wire {annotations = [{class = "circt.tracker", id = distinct[0]<>}]} : !firrtl.uint<8>
    %wire = firrtl.wire : !firrtl.uint<8>
}

firrtl.class @OM() {
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    %0 = firrtl.unresolved_path "OMReferenceTarget:~TargetWire|TargetWire>wire"
}
}

// -----

// CHECK-LABEL: firrtl.circuit "TargetWire"
firrtl.circuit "TargetWire" {
firrtl.module @TargetWire() {
    // CHECK: %wire = firrtl.wire {annotations = [{class = "circt.tracker", id = distinct[0]<>}]} : !firrtl.uint<8>
    %wire = firrtl.wire : !firrtl.uint<8>
    %test0 = firrtl.object @Test0()
    %test1 = firrtl.object @Test1()
}

firrtl.class @Test0(){
  %om = firrtl.object @OM()
}

firrtl.class @Test1() {
  %om = firrtl.object @OM()
}

firrtl.class @OM() {
    // CHECK: %0 = firrtl.path reference distinct[0]<>
    %0 = firrtl.unresolved_path "OMReferenceTarget:~TargetWire|TargetWire>wire"
}
}
