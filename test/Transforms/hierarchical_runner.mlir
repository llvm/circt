// RUN: circt-opt --pass-pipeline='builtin.module(hierarchical-runner{top-name=top pipeline=canonicalize})' %s | FileCheck %s

hw.module @top(out out: i8) {
    %c1_i8 = hw.constant 1 : i8
    %add = comb.add %c1_i8, %c1_i8 : i8
    hw.output %add : i8
}

hw.module @child(out out: i8) {
    %c1_i8 = hw.constant 1 : i8
    %add = comb.add %c1_i8, %c1_i8 : i8
    %0 = hw.instance "bound" @bound() : () -> (){doNotPrint = true} 
    hw.output %add : i8
}

hw.module @bound(out out: i8) {
    %c1_i8 = hw.constant 1 : i8
    %add = comb.add %c1_i8, %c1_i8 : i8
    hw.output %add : i8
}



