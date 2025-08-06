// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{strategy=area test=true max-cuts-per-root=8})' %s | FileCheck %s --check-prefixes CHECK,AREA
// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{strategy=timing test=true max-cuts-per-root=8})' %s | FileCheck %s --check-prefixes CHECK,TIMING

hw.module @and_inv(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
    %0 = aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

hw.module @and_inv_n(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
    %0 = aig.and_inv not %a, %b : i1
    hw.output %0 : i1
}

hw.module @and_inv_nn(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
    %0 = aig.and_inv not %a, not %b : i1
    hw.output %0 : i1
}

// Delay is shorter than @and_inv + @and_inv_n_n. Area is (significantly) larger than @and_inv_n + @and_inv_n_n.
// Check that we use @and_inv_3 if strategy = timing, and @and_inv_n + @and_inv_n_n if strategy = area.
hw.module @and_inv_3(in %a : i1, in %b : i1, in %c : i1, out result : i1) attributes {hw.techlib.info = {area = 10.0 : f64, delay = [[1], [1], [1]]}} {
    %0 = aig.and_inv %a, %b : i1
    %1 = aig.and_inv not %0, %c : i1
    hw.output %1 : i1
}

// CHECK-LABEL: @test_strategy
hw.module @test_strategy(in %a : i1, in %b : i1, in %c : i1, out result : i1) {
    // AREA-NEXT: %[[area_0:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %a: i1, b: %b: i1) -> (result: i1) {test.arrival_times = [1]}
    // AREA-NEXT: %[[area_1:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv_n(a: %[[area_0]]: i1, b: %c: i1) -> (result: i1) {test.arrival_times = [2]}
    // AREA-NEXT: hw.output %[[area_1]] : i1
    // TIMING-NEXT: %[[timing:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv_3(a: %a: i1, b: %b: i1, c: %c: i1) -> (result: i1) {test.arrival_times = [1]}
    // TIMING-NEXT: hw.output %[[timing]] : i1
    %0 = aig.and_inv %a, %b : i1
    %1 = aig.and_inv %c, not %0 : i1
    hw.output %1 : i1
}

hw.module @permutation(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out result: i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1], [1], [1]]}} {
    %0 = aig.and_inv %a, not %b : i1
    %1 = aig.and_inv %c, not %d : i1
    %2 = aig.and_inv %0, not %1 : i1
    hw.output %2 : i1
}

// CHECK-LABEL: hw.module @permutation_test(in %p : i1, in %q : i1, in %r : i1, in %s : i1, out result : i1) {
hw.module @permutation_test(in %p: i1, in %q: i1, in %r: i1, in %s: i1, out result: i1) {
    // {a -> s, b -> p, c -> q, d -> r}
    // CHECK-NEXT: hw.instance "{{.+}}" @permutation(a: %s: i1, b: %p: i1, c: %q: i1, d: %r: i1) -> (result: i1) {test.arrival_times = [1]}
    %0 = aig.and_inv %s, not %p : i1
    %1 = aig.and_inv %q, not %r : i1
    %2 = aig.and_inv %0, not %1 : i1
    hw.output %2 : i1
}

hw.module @and_inv_5(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e: i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [2], [2], [2], [1]]}} {
    %0 = aig.and_inv not %a, %b, not %c, %d, not %e : i1
    hw.output %0 : i1
}

// Make sure truth value is computed correctly for @and_inv_5.
// CHECK-LABEL: @and_inv_5_test
hw.module @and_inv_5_test(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e: i1, out o1 : i1, out o2 : i1) {
    %0 = aig.and_inv not %a, %b : i1
    %1 = aig.and_inv not %c, %d : i1
    %2 = aig.and_inv %0, %1 : i1
    %3 = aig.and_inv %2, not %e : i1
    // CHECK-NEXT: %[[result_0:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv_5(a: %a: i1, b: %b: i1, c: %c: i1, d: %d: i1, e: %e: i1)
    %4 = aig.and_inv not %a, not %d : i1
    %5 = aig.and_inv not %b, %e : i1
    %6 = aig.and_inv %5, %c : i1
    %7 = aig.and_inv %6, %4 : i1
    // CHECK-NEXT: %[[result_1:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv_5(a: %b: i1, b: %e: i1, c: %a: i1, d: %c: i1, e: %d: i1)
    
    hw.output %3, %7 : i1, i1
    // CHECK-NEXT: hw.output %[[result_0]], %[[result_1]] : i1, i1
}

hw.module @area_flow(in %a : i1, in %b : i1, in %c: i1, out result : i1) attributes {hw.techlib.info = {area = 1.5 : f64, delay = [[10], [10], [10], [10], [10]]}} {
    %0 = aig.and_inv not %a, not %b : i1
    %1 = aig.and_inv not %c, %0 : i1
    hw.output %1 : i1
}

// This is a test that needs area-flow to get an optimal result.
// It produces sub-optimal mappings since currently area-flow is not implemented.
// CHECK-LABEL: @area_flow_test
hw.module @area_flow_test(in %a : i1, in %b : i1, in %c: i1, out result : i1) {
    // FIXME: If area-flow is implemented, this should be mapped to @area_flow with area strategy.
    // CHECK:       hw.instance {{.*}} @and_inv_nn(
    // CHECK-NEXT:  hw.instance {{.*}} @and_inv_n(
    %0 = aig.and_inv not %a, not %b : i1
    %1 = aig.and_inv not %c, %0 : i1
    hw.output %1 : i1
}

// Test primary inputs handling
// CHECK-LABEL: @primary_inputs_test
hw.module @primary_inputs_test(in %a : i1, in %b : i1, out result : i1) {
    // Simple direct mapping - should use @and_inv
    // CHECK-NEXT: %[[primary:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %a: i1, b: %b: i1) -> (result: i1) {test.arrival_times = [1]}
    // CHECK-NEXT: hw.output %[[primary]] : i1
    %0 = aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

// Test chain of operations for timing analysis
// CHECK-LABEL: @timing_chain_test
hw.module @timing_chain_test(in %a : i1, in %b : i1, in %c : i1, in %d : i1, out result : i1) {
    // Test that timing is accumulated correctly through the chain
    // CHECK: hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %a: i1, b: %b: i1) -> (result: i1) {test.arrival_times = [1]}
    // CHECK: hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %c: i1, b: %d: i1) -> (result: i1) {test.arrival_times = [1]}
    // CHECK: hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %{{.+}}: i1, b: %{{.+}}: i1) -> (result: i1) {test.arrival_times = [2]}
    %0 = aig.and_inv %a, %b : i1
    %1 = aig.and_inv %c, %d : i1
    %2 = aig.and_inv %0, %1 : i1
    hw.output %2 : i1
}

// Test comb.extract and comb.concat handling
// CHECK-LABEL: @extract_concat_test
hw.module @extract_concat_test(in %data : i4, in %ctrl : i2, out result : i3) {
    // Extract individual bits from multi-bit inputs
    %bit0 = comb.extract %data from 0 : (i4) -> i1
    %bit1 = comb.extract %data from 1 : (i4) -> i1
    %bit2 = comb.extract %data from 2 : (i4) -> i1
    %bit3 = comb.extract %data from 3 : (i4) -> i1
    
    %ctrl0 = comb.extract %ctrl from 0 : (i2) -> i1
    %ctrl1 = comb.extract %ctrl from 1 : (i2) -> i1
    
    // Apply some logic using AIG operations
    %and0 = aig.and_inv %bit0, %bit1 : i1
    %and1 = aig.and_inv %bit2, not %ctrl0 : i1
    %and2 = aig.and_inv %bit3, %ctrl1 : i1
    
    // Further logic operations
    %out0 = aig.and_inv %and0, %ctrl0 : i1
    %out1 = aig.and_inv %and1, not %and0 : i1
    %out2 = aig.and_inv %and2, %ctrl1 : i1
    
    // Concatenate results into multi-bit output
    %result_concat = comb.concat %out2, %out1, %out0 : i1, i1, i1
    
    // CHECK:      %[[bit0:.+]] = comb.extract %data from 0
    // CHECK-NEXT: %[[bit1:.+]] = comb.extract %data from 1
    // CHECK-NEXT: %[[bit2:.+]] = comb.extract %data from 2
    // CHECK-NEXT: %[[bit3:.+]] = comb.extract %data from 3
    // CHECK-NEXT: %[[ctrl0:.+]] = comb.extract %ctrl from 0
    // CHECK-NEXT: %[[ctrl1:.+]] = comb.extract %ctrl from 1
    // CHECK-NEXT: %[[and0:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %[[bit0]]: i1, b: %[[bit1]]: i1) -> (result: i1) {test.arrival_times = [1]}
    // CHECK-NEXT: %[[and1:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv_n(a: %[[ctrl0]]: i1, b: %[[bit2]]: i1) -> (result: i1) {test.arrival_times = [1]}
    // CHECK-NEXT: %[[out0:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %[[and0]]: i1, b: %[[ctrl0]]: i1) -> (result: i1) {test.arrival_times = [2]}
    // CHECK-NEXT: %[[out1:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv_n(a: %[[and0]]: i1, b: %[[and1]]: i1) -> (result: i1) {test.arrival_times = [2]}
    // CHECK-NEXT: %[[out2:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %[[bit3]]: i1, b: %[[ctrl1]]: i1) -> (result: i1) {test.arrival_times = [1]}
    // CHECK-NEXT: %[[result:.+]] = comb.concat %[[out2]], %[[out1]], %[[out0]] : i1, i1, i1
    // CHECK-NEXT: hw.output %[[result]] : i3
    
    hw.output %result_concat : i3
}
