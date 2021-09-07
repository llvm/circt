// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{Invalid LLHD type!}}
func @illegaltype(%arg0: !llhd.illegaltype) {
    return
}

// -----

// expected-error @+2 {{Invalid LLHD attribute!}}
func @illegalattr() {
    %0 = llhd.constant_time #llhd.illegalattr : i1
    return
}
