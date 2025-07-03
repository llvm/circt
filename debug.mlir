// CHECK-LABEL: module ArrayInject(
hw.module @ArrayInject(in %a: !hw.array<4xi42>, in %b: i42, in %i: i2, out z: !hw.array<4xi42>) {
  // CHECK: reg [3:0][41:0] [[TMP:.+]];
  // CHECK-NEXT: always_comb begin
  // CHECK-NEXT:   [[TMP]] = a;
  // CHECK-NEXT:   [[TMP]][i] = b;
  // CHECK-NEXT: end
  %0 = hw.array_inject %a[%i], %b : !hw.array<4xi42>, i2
  // CHECK-NEXT: assign z = [[TMP]];
  hw.output %0 : !hw.array<4xi42>
}
