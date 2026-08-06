// RUN: circt-opt %s --allow-unregistered-dialect --split-input-file --verify-diagnostics

// expected-error @below {{expected ','}}
"test.attrs"() {a = #axi4.burst_spec<fixed>} : () -> ()

// -----

// expected-error @below {{'fixed' burst 'len' must be between 1 and 16, got 0}}
"test.attrs"() {a = #axi4.burst_spec<fixed, len = 0>} : () -> ()

// -----

// expected-error @below {{'fixed' burst 'len' must be between 1 and 16, got 17}}
"test.attrs"() {a = #axi4.burst_spec<fixed, len = 17>} : () -> ()

// -----

// expected-error @below {{'incr' burst 'len' must be between 1 and 256, got 0}}
"test.attrs"() {a = #axi4.burst_spec<incr, len = 0>} : () -> ()

// -----

// expected-error @below {{'incr' burst 'len' must be between 1 and 256, got 257}}
"test.attrs"() {a = #axi4.burst_spec<incr, len = 257>} : () -> ()

// -----

// expected-error @below {{'wrap' burst 'len' must be 2, 4, 8, or 16, got 1}}
"test.attrs"() {a = #axi4.burst_spec<wrap, len = 1>} : () -> ()

// -----

// expected-error @below {{'wrap' burst 'len' must be 2, 4, 8, or 16, got 7}}
"test.attrs"() {a = #axi4.burst_spec<wrap, len = 7>} : () -> ()

// -----

// expected-error @below {{'wrap' burst 'len' must be 2, 4, 8, or 16, got 32}}
"test.attrs"() {a = #axi4.burst_spec<wrap, len = 32>} : () -> ()

// -----

// expected-error @below {{'burst_set' must be non-empty}}
"test.attrs"() {a = #axi4.burst_set<>} : () -> ()
