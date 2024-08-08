// RUN: circt-opt --verify-diagnostics --split-input-file %s

// expected-error @below {{integer literal requires at least 6 bits, but attribute specifies only 3}}
hw.constant false {foo = #moore.fvint<42 : 3>}
