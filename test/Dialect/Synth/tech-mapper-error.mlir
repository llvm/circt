// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper)' %s --split-input-file --verify-diagnostics

hw.module @do_nothing(in %a : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1]]}} {
    hw.output %a : i1
}

hw.module @test(in %a : i1, in %b : i1, out result : i1) {
    // expected-error-re@+1 {{No matching cut found for value: {{.*}}}}
    %0 = synth.aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

// -----

hw.module @do_nothing(in %a : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1]]}} {
    hw.output %a : i1
}

// Test for too many operands in AIG operation
hw.module @test(in %a : i1, in %b : i1, out result : i1) {
    // expected-error-re@+1 {{Cut enumeration supports at most 2 operands, found: 3}}
    %0 = synth.aig.and_inv %a, %b, %a : i1
    hw.output %0 : i1
}


// -----

// expected-error@+1 {{Modules with multiple outputs are not supported yet}}
hw.module @multi_output(in %a : i1, out result1 : i1, out result2 : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
    hw.output %a, %a : i1, i1
}

hw.module @test(in %a : i1, out result : i1) {
    hw.output %a : i1
}


// -----

// expected-error@+1 {{All input ports must be single bit}}
hw.module @multibit(in %a : i2, in %b: i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [2]]}} {
  hw.output %b: i1
}

// -----


// expected-error@+1 {{All output ports must be single bit}}
hw.module @multibit(in %a : i1, in %b: i1, out result : i2) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [2]]}} {
  %0 = comb.concat %a, %b : i1, i1
  hw.output %0: i2
}

// -----


hw.module @multibit(in %a : i1, in %b: i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [2]]}} {
  // expected-error@+1 {{Unsupported operation for truth table simulation}}
  %0 = comb.xor %a, %b : i1
  hw.output %0: i1
}

// -----

// expected-error@+1 {{Too many inputs for truth table generation}}
hw.module @too_many_input_bits(in %a0 : i1, in %a1 : i1, in %a2 : i1, in %a3 : i1, in %a4 : i1,
                               in %a5 : i1, in %a6 : i1, in %a7 : i1, in %a8 : i1, in %a9 : i1,
                               in %a10 : i1, in %a11 : i1, in %a12 : i1, in %a13 : i1, in %a14 : i1,
                               in %a15 : i1, in %a16 : i1,  out result : i1)
  attributes {hw.techlib.info = {
  area = 1.0 : f64,
  delay = [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
           [1], [1], [1], [1], [1], [1], [1]]
  }} {
  %0 = synth.aig.and_inv %a0, %a1 : i1
  hw.output %0: i1
}
