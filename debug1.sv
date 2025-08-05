timeunit 100ps / 1ps;

module Foo;
  realtime t0 = 1;
  realtime t1 = 2ns;
  realtime t2 = 0.3ns;
  realtime t3 = 0.1234567ns;
  realtime t4 = 45.678ps;
  realtime t5 = 34567fs;

  initial begin
    Foo.magic(t0);
    Bar.magic(t0);
  end

  task magic(realtime t);
  endtask
endmodule

