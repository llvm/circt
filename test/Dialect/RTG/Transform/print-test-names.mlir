// RUN: circt-opt %s --rtg-print-test-names=filename=%t/out.csv && FileCheck %s --input-file=%t/out.csv

// CHECK: test1_config1,test1{{$}}
// CHECK: test1_config2,test1{{$}}
// CHECK: test3_config3,test3{{$}}
// CHECK: test_without_template,test_without_template{{$}}

rtg.test @test1_config1() template "test1" { }
rtg.test @test1_config2() template "test1" { }
rtg.test @test3_config3() template "test3" { }
rtg.test @test_without_template() {}
