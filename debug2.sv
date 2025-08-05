timeunit 1ns / 1ps;

module Bar;
  realtime t0 = 2ns;

  initial begin
    Foo.magic(t0);
    Bar.magic(t0);
  end

  task magic(realtime t);
  endtask
endmodule
