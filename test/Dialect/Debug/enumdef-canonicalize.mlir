// RUN: circt-opt --canonicalize %s | FileCheck %s

module {
  func.func @Test() {
    // CHECK: %[[E:.+]] = dbg.enumdef "S", fqn "p.S", {A = 0 : i64}
    // CHECK-NOT: = dbg.enumdef "S", fqn "p.S", {A = 0 : i64}
    %e0 = dbg.enumdef "S", fqn "p.S", {A = 0 : i64}
    %e1 = dbg.enumdef "S", fqn "p.S", {A = 0 : i64}

    %0 = arith.constant 0 : i2
    // CHECK: dbg.variable "x", {{.*}} enumDef %[[E]]
    // CHECK: dbg.variable "y", {{.*}} enumDef %[[E]]
    dbg.variable "x", %0 enumDef %e0 : i2
    dbg.variable "y", %0 enumDef %e1 : i2
    return
  }

  // CHECK-LABEL: func @TestDivergingVariants
  // Test that enumdefs with same fqn but different variants are NOT deduplicated
  // Both should remain since they have different semantics
  func.func @TestDivergingVariants() {
    // CHECK: %[[E0:.+]] = dbg.enumdef "S", fqn "p.S", {A = 0 : i64}
    // CHECK: %[[E1:.+]] = dbg.enumdef "S", fqn "p.S", {A = 0 : i64, B = 1 : i64}
    // both enumdefs should remain (different variants)
    %e0 = dbg.enumdef "S", fqn "p.S", {A = 0 : i64}
    %e1 = dbg.enumdef "S", fqn "p.S", {A = 0 : i64, B = 1 : i64}

    %0 = arith.constant 0 : i2
    // CHECK: dbg.variable "x", {{.*}} enumDef %[[E0]]
    // CHECK: dbg.variable "y", {{.*}} enumDef %[[E1]]
    dbg.variable "x", %0 enumDef %e0 : i2
    dbg.variable "y", %0 enumDef %e1 : i2
    return
  }

  // Scope-sensitive dedup: two enumdefs describing the same source-level enum
  // type but living in different inline scopes must NOT be merged, because
  // their SSA tokens are consumed by variables that belong to those distinct
  // scopes. See the EnumDefDeduplication rationale.
  //
  // CHECK-LABEL: func @TestScopeDistinct
  func.func @TestScopeDistinct() {
    // CHECK: %[[S1:.+]] = dbg.scope "a", "A"
    // CHECK: %[[S2:.+]] = dbg.scope "b", "B"
    %s1 = dbg.scope "a", "A"
    %s2 = dbg.scope "b", "B"
    // CHECK: %[[E0:.+]] = dbg.enumdef "S", fqn "p.S", {A = 0 : i64} scope %[[S1]]
    // CHECK: %[[E1:.+]] = dbg.enumdef "S", fqn "p.S", {A = 0 : i64} scope %[[S2]]
    %e0 = dbg.enumdef "S", fqn "p.S", {A = 0 : i64} scope %s1
    %e1 = dbg.enumdef "S", fqn "p.S", {A = 0 : i64} scope %s2

    %0 = arith.constant 0 : i2
    // CHECK: dbg.variable "x", {{.*}} enumDef %[[E0]]
    // CHECK: dbg.variable "y", {{.*}} enumDef %[[E1]]
    dbg.variable "x", %0 enumDef %e0 : i2
    dbg.variable "y", %0 enumDef %e1 : i2
    return
  }

  // Within a single shared scope, two enumdefs with identical (fqn, variants)
  // SHOULD still deduplicate.
  //
  // CHECK-LABEL: func @TestScopeSameDedups
  func.func @TestScopeSameDedups() {
    // CHECK: %[[S:.+]] = dbg.scope "a", "A"
    %s = dbg.scope "a", "A"
    // CHECK: %[[E:.+]] = dbg.enumdef "S", fqn "p.S", {A = 0 : i64} scope %[[S]]
    // CHECK-NOT: = dbg.enumdef "S", fqn "p.S", {A = 0 : i64} scope
    %e0 = dbg.enumdef "S", fqn "p.S", {A = 0 : i64} scope %s
    %e1 = dbg.enumdef "S", fqn "p.S", {A = 0 : i64} scope %s

    %0 = arith.constant 0 : i2
    // CHECK: dbg.variable "x", {{.*}} enumDef %[[E]]
    // CHECK: dbg.variable "y", {{.*}} enumDef %[[E]]
    dbg.variable "x", %0 enumDef %e0 : i2
    dbg.variable "y", %0 enumDef %e1 : i2
    return
  }

  // Mixed: one enumdef at module-scope (scope=null), one under a dbg.scope;
  // both must remain because a null scope is distinct from any non-null scope.
  //
  // CHECK-LABEL: func @TestScopeNullVsNonNull
  func.func @TestScopeNullVsNonNull() {
    // CHECK: %[[S:.+]] = dbg.scope "a", "A"
    %s = dbg.scope "a", "A"
    // CHECK: %[[E0:.+]] = dbg.enumdef "S", fqn "p.S", {A = 0 : i64}
    // CHECK-NOT: scope %{{.*}}
    // CHECK: %[[E1:.+]] = dbg.enumdef "S", fqn "p.S", {A = 0 : i64} scope %[[S]]
    %e0 = dbg.enumdef "S", fqn "p.S", {A = 0 : i64}
    %e1 = dbg.enumdef "S", fqn "p.S", {A = 0 : i64} scope %s

    %0 = arith.constant 0 : i2
    // CHECK: dbg.variable "x", {{.*}} enumDef %[[E0]]
    // CHECK: dbg.variable "y", {{.*}} enumDef %[[E1]]
    dbg.variable "x", %0 enumDef %e0 : i2
    dbg.variable "y", %0 enumDef %e1 : i2
    return
  }
}
