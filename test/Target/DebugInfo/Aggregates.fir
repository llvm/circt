; RUN: true
FIRRTL version 3.5.0

circuit Top:
  module Top:
    input bus: {
      valid: UInt<1>,
      flip ready: UInt<1>,
      args: {lhs: SInt<19>, rhs: SInt<19>},
      flip result: {left: SInt<20>, right: SInt<20>}
    }

    inst alu of ALU
    connect alu.disable, UInt<1>(0)
    connect alu.args, bus.args
    connect bus.result, alu.result.data
    connect bus.ready, alu.result.is_available

  module ALU:
    input args: {lhs: SInt<19>, rhs: SInt<19>}
    input disable: UInt<1>
    output result: {data: {left: SInt<20>, right: SInt<20>}, is_available: UInt<1>}

    node tmpAdd = add(args.lhs, args.rhs)
    node tmpSub = sub(args.lhs, args.rhs)

    connect result.data.left, tmpAdd
    connect result.data.right, tmpSub
    connect result.is_available, UInt<1>(1)

    when disable:
      connect result.data.left, SInt(0)
      connect result.data.right, SInt(0)
