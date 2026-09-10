// RUN: circt-opt %s -split-input-file --verify-diagnostics

hw.module.extern @Foo (in %in0: i1, out out: i1)
esi.pure_module @top {
  // expected-error @+1 {{'hw.instance' op instances in ESI pure modules can only contain channel ports}}
  %loopback = hw.instance "foo" @Foo(in0: %loopback: i1) -> (out: i1)
}

// -----

hw.module.extern @Foo (out out: i1)

esi.pure_module @top {
  // expected-error @+1 {{'hw.instance' op instances in ESI pure modules can only contain channel ports}}
  hw.instance "foo" @Foo() -> (out: i1)
}

// -----

esi.pure_module @top {
  %loopback = esi.null : !esi.channel<i1>
  %data, %valid = esi.unwrap.vr %loopback, %ready : i1
  // expected-error @+1 {{'comb.add' op operation not allowed in ESI pure modules}}
  %double = comb.add %data, %data : i1
  %loopbackDouble, %ready = esi.wrap.vr %double, %valid : i1
}

// -----

esi.pure_module @top {
  // expected-note @+1 {{}}
  %a0 = esi.pure_module.input "a" : i1
  // expected-error @+1 {{port 'a' previously declared as type 'i1'}}
  %a1 = esi.pure_module.input "a" : i5
}

// -----

esi.pure_module @top {
  // expected-note @+1 {{}}
  %a0 = esi.pure_module.input "a" : i1
  // expected-error @+1 {{port 'a' previously declared}}
  esi.pure_module.output "a", %a0 : i1
}
