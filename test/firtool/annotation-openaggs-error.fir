; RUN: firtool -parse-only %s --verify-diagnostics

FIRRTL version 4.0.0
; expected-error @below {{Unable to resolve target of annotation: {class = "circt.test", target = "~OpenAgg|OpenAgg>out.p"}}
circuit OpenAgg : %[[
 { "class": "circt.test",
   "target": "~OpenAgg|OpenAgg>out.p"
 }
]]
  ; expected-error @below {{cannot resolve field 'p'}}
  public module OpenAgg:
    output out : {a : UInt<1>,
                  p : Probe<UInt<1>>,
                  b : UInt<1>}

    connect out.a, UInt<1>(1)
    connect out.b, UInt<1>(0)

    node n = out.a
    define out.p = probe(n)
