import chisel3._

class Top(val num: Int) extends RawModule {
  val in = IO(Input(UInt(16.W)))
  val out = IO(Output(UInt(16.W)))
  var last = in
  for (i <- 0 until num) {
    val a = new Alice(i, this)  // inlined
    a.b.in := last
    last = a.b.out
  }
  out := last
  def outOfLineFn(o : Data, i : Data) = { o := i }
}

class Alice(val index: Int, val ex : Top) {
  val b = Module(new Bob(ex));
  b.suggestName("b"+index)
}

class Bob(val ex : Top) extends RawModule {
  val in = IO(Input(UInt(16.W)))
  val out = IO(Output(UInt(16.W)))
  val x = ~in
  ex.outOfLineFn(out, x)
}

println((new chisel3.stage.ChiselStage).emitChirrtl(new Top(2)))
