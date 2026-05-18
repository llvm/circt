// RUN: circt-opt --verify-roundtrip %s | FileCheck %s

module {
  func.func @Test() {
    %e = dbg.enumdef "FieldEnum", fqn "pkg.FieldEnum", {VALUE = 42 : i64}
    %0 = arith.constant 0 : i32

    // CHECK: %[[SF:.+]] = dbg.subfield "field", %{{.*}} enumDef %{{.*}} : i32
    %f = dbg.subfield "field", %0 enumDef %e : i32

    // Use in a struct so SubFieldOp is not treated as dead code.
    %s = dbg.struct {"field": %f} : !dbg.subfield
    return
  }
}
