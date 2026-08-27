// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @illegal_net_element() {
  // expected-error @+1 {{invalid element for sv.net type}}
  %x = sv.wire : !sv.net<!hw.inout<i1>>
}

// -----

hw.module @illegal_var_element() {
  // expected-error @+1 {{invalid element for sv.var type}}
  %x = sv.var : !sv.var<!hw.inout<i1>>
}

// -----

hw.module @illegal_assign(in %input : i1) {
  // expected-note @+1 {{prior use here}}
  %var = sv.var : !sv.var<i1>
  // expected-error @+1 {{use of value '%var' expects different type than prior uses: '!sv.net<i1>' vs '!sv.var<i1>'}}
  sv.assign %var, %input : i1
}

// -----

hw.module @illegal_procedural_assign(in %input : i1) {
  // expected-note @+1 {{prior use here}}
  %net = sv.wire : !sv.net<i1>
  sv.alwayscomb {
    // expected-error @+1 {{use of value '%net' expects different type than prior uses: '!sv.var<i1>' vs '!sv.net<i1>'}}
    sv.bpassign %net, %input : i1
  }
}

// -----

hw.module @illegal_alias() {
  %net = sv.wire : !sv.net<i1>
  %var = sv.var : !sv.var<i1>
  // expected-error @+1 {{'sv.alias' op operand #1 must be variadic of an SV net type, but got '!sv.var<i1>'}}
  sv.alias %net, %var : !sv.net<i1>, !sv.var<i1>
}

// -----

hw.module @illegal_force(in %input : i1, in %index : i1) {
  %array = sv.var : !sv.var<!hw.array<2xi1>>
  %element = sv.array_index_inout %array[%index] : !sv.var<!hw.array<2xi1>>, i1
  sv.alwayscomb {
    // expected-error @+1 {{cannot force a memory word or bit/part-select of a variable}}
    sv.force %element, %input : !sv.var<i1>
  }
}

// -----

hw.module @illegal_readmem() {
  %net = sv.wire : !sv.net<!hw.uarray<8xi32>>
  sv.initial {
    // expected-error @+1 {{'sv.readmem' op operand #0 must be an SV variable type, but got '!sv.net<!hw.uarray<8xi32>>'}}
    sv.readmem %net, "memory.hex", MemBaseHex : !sv.net<!hw.uarray<8xi32>>
  }
}
