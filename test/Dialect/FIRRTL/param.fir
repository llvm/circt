circuit MyModule :

  module ParamModule2 <
    parameter input someparam : UInt<5>
    parameter output something : UInt<5>
// Other ports
  >
// Parameter Resolution
  connect something, someparam
// Logic Level
  // Stuff ...

  extmodule IndModule <
    parameter input inP : UInt
    input in : UInt
    output out : UInt
  >

  module ParamModule <
    parameter input inParam : UInt
    parameter output outParam : UInt
    parameter input inTypeParam : Type
    parameter input inModParam : Module<parameter input inP : UInt, input in : UInt, output out : UInt>
    input inVal : inTypeParam
    output outVal : Uint<inParam>
  >
// Param Resolution
    node foo = add(inParam, UInt(5))
    resolve resolvedMod of ParamModule2 with foo  
    resolve resolvedMod2 of ParamModule2 with resolvedMod.something // note the explicit query of a computed type
    connect outParam, resolvedMod.something // note the explicit query of a computed type

// Logic Level
    inst m1 of resolvedMod
    inst m2 of inModParam
    wire w : inTypeParam[3]
    // stuff ...



  module MyModule : 
    input in: UInt
    output out: UInt<8>
// Param Resolution
    resolve mod of ParamModule with UInt(5), Clock, IndModule
// Logic Level
  inst m of mod
  // stuff ...

