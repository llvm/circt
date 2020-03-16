// expected-error {{firrtl parsing not implemented}}

// RUN: spt-translate -import-firrtl -verify-diagnostics %s 

module MyModule :
  input in: UInt<8>
  output out: UInt<8>
  out <= in
