// RUN: circt-opt --verify-roundtrip %s | FileCheck %s

module {
  func.func @Test() {
    // CHECK: %[[E:.+]] = dbg.enumdef "MyState", fqn "pkg.MyState", {A = 0 : i64, B = 1 : i64}
    %e = dbg.enumdef "MyState", fqn "pkg.MyState", {A = 0 : i64, B = 1 : i64}

    %0 = arith.constant 0 : i2
    // CHECK: dbg.variable "state", {{.+}} enumDef %[[E]] : i2
    dbg.variable "state", %0 enumDef %e : i2
    return
  }
}
