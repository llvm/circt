// RUN: firld %s --base-circuit Outer | FileCheck %s

module {
  firrtl.circuit "Outer" {
    // CHECK: firrtl.class private @Outer_InnerOM
    firrtl.class private @InnerOM(out %width: !firrtl.integer) {
      %0 = firrtl.integer 16
      firrtl.propassign %width, %0 : !firrtl.integer
    }
    // CHECK: firrtl.module private @Outer_Inner(in %in: !firrtl.uint<16>, out %out: !firrtl.uint<16>, out %om: !firrtl.class<@Outer_InnerOM(out width: !firrtl.integer)>) {
    firrtl.module private @Inner(in %in: !firrtl.uint<16>, out %out: !firrtl.uint<16>, out %om: !firrtl.class<@InnerOM(out width: !firrtl.integer)>) {
      firrtl.matchingconnect %out, %in : !firrtl.uint<16>
      // CHECK: %omInstance = firrtl.object @Outer_InnerOM(out width: !firrtl.integer)
      %omInstance = firrtl.object @InnerOM(out width: !firrtl.integer)
      // CHECK: firrtl.propassign %om, %omInstance : !firrtl.class<@Outer_InnerOM(out width: !firrtl.integer)>
      firrtl.propassign %om, %omInstance : !firrtl.class<@InnerOM(out width: !firrtl.integer)>
    }
    firrtl.module @Outer() attributes {convention = #firrtl<convention scalarized>} {
      // CHECK: %Inner_in, %Inner_out, %Inner_om = firrtl.instance Inner interesting_name @Outer_Inner(in in: !firrtl.uint<16>, out out: !firrtl.uint<16>, out om: !firrtl.class<@Outer_InnerOM(out width: !firrtl.integer)>)
      %Inner_in, %Inner_out, %Inner_om = firrtl.instance Inner interesting_name @Inner(in in: !firrtl.uint<16>, out out: !firrtl.uint<16>, out om: !firrtl.class<@InnerOM(out width: !firrtl.integer)>)
    }
  }
}
