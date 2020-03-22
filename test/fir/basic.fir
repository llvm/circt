; RUN: spt-translate -import-firrtl -verify-diagnostics %s

circuit basic :
  module MyModule :
    input in: UInt<8>
    output out: UInt<8>
    out <= in


  ; Module with a keyword id.
  module circuit :
    input in: UInt<8>

