// RUN: circt-opt %s --moore-vectorization -allow-unregistered-dialect | FileCheck %s

//===----------------------------------------------------------------------===//
// Test 1: "Happy Path". All 4 bits are present and contiguous.
// The pass SHOULD vectorize this.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: moore.module @simple_vectorization
module {
  moore.module @simple_vectorization(out out : !moore.l4, in %in : !moore.l4) {
    %out_wire = moore.net wire : <l4>

    // CHECK-NOT: moore.assign {{.*}} : l1
    // CHECK:     moore.assign %out_wire, %in : l4
    
    %0 = moore.extract_ref %out_wire from 3 : <l4> -> <l1>
    %1 = moore.extract %in from 3 : l4 -> l1
    moore.assign %0, %1 : l1
    
    %2 = moore.extract_ref %out_wire from 2 : <l4> -> <l1>
    %3 = moore.extract %in from 2 : l4 -> l1
    moore.assign %2, %3 : l1
    
    %4 = moore.extract_ref %out_wire from 1 : <l4> -> <l1>
    %5 = moore.extract %in from 1 : l4 -> l1
    moore.assign %4, %5 : l1
    
    %6 = moore.extract_ref %out_wire from 0 : <l4> -> <l1>
    %7 = moore.extract %in from 0 : l4 -> l1
    moore.assign %6, %7 : l1
    
    %8 = moore.read %out_wire : <l4>
    moore.output %8 : !moore.l4
  }
}

//===----------------------------------------------------------------------===//
// Test 2: Partial vectorization case.
// The contiguous group [0, 1] has size 2, but the vector has size 4.
// The pass should NOT vectorize this. The output should be the same as the input.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: moore.module @test_partial_vectorization
moore.module @test_partial_vectorization(in %in : !moore.l4, out out: !moore.l4) {
  %out_wire = moore.net wire : <l4>

  // CHECK: %[[S0:.+]] = moore.extract %in from 0
  // CHECK: moore.assign %{{.*}}, %[[S0]]
  // CHECK: %[[S1:.+]] = moore.extract %in from 1
  // CHECK: moore.assign %{{.*}}, %[[S1]]
  // CHECK: %[[S3:.+]] = moore.extract %in from 3
  // CHECK: moore.assign %{{.*}}, %[[S3]]

  %s0 = moore.extract %in from 0 : l4 -> l1
  %d0 = moore.extract_ref %out_wire from 0 : <l4> -> <l1>
  moore.assign %d0, %s0 : l1

  %s1 = moore.extract %in from 1 : l4 -> l1
  %d1 = moore.extract_ref %out_wire from 1 : <l4> -> <l1>
  moore.assign %d1, %s1 : l1

  %s3 = moore.extract %in from 3 : l4 -> l1
  %d3 = moore.extract_ref %out_wire from 3 : <l4> -> <l1>
  moore.assign %d3, %s3 : l1

  %8 = moore.read %out_wire : <l4>
  moore.output %8 : !moore.l4
}

//===----------------------------------------------------------------------===//
// Test 3: Negative case. An extracted bit has multiple uses.
//         The pass should do nothing thanks to the `hasOneUse` check.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: moore.module @test_no_vectorize_on_multi_use
moore.module @test_no_vectorize_on_multi_use(in %in : !moore.l2, out out: !moore.l2) {
  %out_wire = moore.net wire : <l2>

  // CHECK: %[[S0:.+]] = moore.extract %in from 0
  // CHECK: "another.op"(%[[S0]])
  // CHECK: moore.assign %{{.*}}, %[[S0]]
  // CHECK: %[[S1:.+]] = moore.extract %in from 1
  // CHECK: moore.assign %{{.*}}, %[[S1]]
  
  %s0 = moore.extract %in from 0 : l2 -> l1
  %d0 = moore.extract_ref %out_wire from 0 : <l2> -> <l1>
  "another.op"(%s0) : (!moore.l1) -> ()
  moore.assign %d0, %s0 : l1

  %s1 = moore.extract %in from 1 : l2 -> l1
  %d1 = moore.extract_ref %out_wire from 1 : <l2> -> <l1>
  moore.assign %d1, %s1 : l1

  %8 = moore.read %out_wire : <l2>
  moore.output %8 : !moore.l2
}

//===----------------------------------------------------------------------===//
// Test 4: "Happy Path" with out-of-order assignments.
// The pass should sort the indices and still perform a full vectorization.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: moore.module @test_out_of_order_vectorization
moore.module @test_out_of_order_vectorization(in %in : !moore.l8, out out: !moore.l8) {
  %out_wire = moore.net wire : <l8>

  // CHECK-NOT: moore.assign {{.*}} : l1
  // CHECK:     moore.assign %out_wire, %in : l8

  %s7 = moore.extract %in from 7 : l8 -> l1
  %d7 = moore.extract_ref %out_wire from 7 : <l8> -> <l1>
  moore.assign %d7, %s7 : l1
  
  %s2 = moore.extract %in from 2 : l8 -> l1
  %d2 = moore.extract_ref %out_wire from 2 : <l8> -> <l1>
  moore.assign %d2, %s2 : l1
  
  %s6 = moore.extract %in from 6 : l8 -> l1
  %d6 = moore.extract_ref %out_wire from 6 : <l8> -> <l1>
  moore.assign %d6, %s6 : l1
  
  %s1 = moore.extract %in from 1 : l8 -> l1
  %d1 = moore.extract_ref %out_wire from 1 : <l8> -> <l1>
  moore.assign %d1, %s1 : l1
  
  %s5 = moore.extract %in from 5 : l8 -> l1
  %d5 = moore.extract_ref %out_wire from 5 : <l8> -> <l1>
  moore.assign %d5, %s5 : l1
  
  %s3 = moore.extract %in from 3 : l8 -> l1
  %d3 = moore.extract_ref %out_wire from 3 : <l8> -> <l1>
  moore.assign %d3, %s3 : l1
  
  %s4 = moore.extract %in from 4 : l8 -> l1
  %d4 = moore.extract_ref %out_wire from 4 : <l8> -> <l1>
  moore.assign %d4, %s4 : l1
  
  %s0 = moore.extract %in from 0 : l8 -> l1
  %d0 = moore.extract_ref %out_wire from 0 : <l8> -> <l1>
  moore.assign %d0, %s0 : l1

  %r = moore.read %out_wire : <l8>
  moore.output %r : !moore.l8
}