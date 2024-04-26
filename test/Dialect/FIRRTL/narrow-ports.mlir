// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-narrow-ports))' %s | FileCheck %s

firrtl.circuit "TopLevel" {

  firrtl.module private @Simple(in %source: !firrtl.uint<42>, out %dest : !firrtl.uint<7>) {
      %narrow = firrtl.bits %source 6 to 0 : (!firrtl.uint<42>) -> !firrtl.uint<7>
      firrtl.strictconnect %dest, %narrow : !firrtl.uint<7>
  }

  firrtl.module @TopLevel(in %source: !firrtl.uint<42>, out %dest : !firrtl.uint<7>) {
    %sourceV, %sinkV = firrtl.instance "" @Simple(in source: !firrtl.uint<42>, out dest : !firrtl.uint<7>)
    firrtl.strictconnect %dest, %sinkV : !firrtl.uint<7>
    firrtl.strictconnect %sourceV, %source : !firrtl.uint<42>
  }


}
