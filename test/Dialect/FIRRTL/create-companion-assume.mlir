    
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-create-companion-assume)))' %s | FileCheck %s
firrtl.circuit "Assert" {

// CHECK-LABEL: firrtl.module @Assert
firrtl.module @Assert(in %clock: !firrtl.clock, in %pred: !firrtl.uint<1>,  in %en: !firrtl.uint<1>,
                      in %value: !firrtl.uint<42>) {
    firrtl.assert %clock, %pred, %en, "assert0"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42> {guards=["Foo"]}
    // CHECK: firrtl.assert
    // CHECK-NEXT: firrtl.assume %clock, %pred, %en, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-SAME: {eventControl = 0 : i32, guards = ["USE_PROPERTY_AS_CONSTRAINT", "Foo"], isConcurrent = true}

    firrtl.assert %clock, %pred, %en, "assert0"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42> {guards=["USE_UNR_ONLY_CONSTRAINTS"]}
    // CHECK: firrtl.assert
    // CHECK-NEXT: firrtl.int.unclocked_assume %pred, %en, "" : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-SAME: {guards = ["USE_PROPERTY_AS_CONSTRAINT", "USE_UNR_ONLY_CONSTRAINTS"]}
}

}
