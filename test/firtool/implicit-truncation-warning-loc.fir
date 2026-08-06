; RUN: firtool %s --format=fir --verify-diagnostics

FIRRTL version 4.0.0
circuit TruncWarn:
  public module TruncWarn:
    input sel : {|A, B|}
    input in : UInt<7>
    output out : UInt<4>

    match sel:
      A:
        ; expected-warning@+1 {{inserting implicit truncation}}
        connect out, in
      B:
        connect out, UInt<4>(0h0)
        