// RUN: querytool %s --filter "**" | FileCheck --check-prefix=TEST1 %s
// TEST1: ::test
// TEST1: ::test::nya
// TEST1: ::test::nya::uwu
// TEST1: ::test::owo
// TEST1: ::nya
// TEST1: ::nya::uwu
// TEST1: ::uwu
// TEST1: ::owo

hw.module @owo() -> () { }
hw.module @uwu() -> () { }

hw.module @nya() -> () {
  hw.instance "uwu1" @uwu() : () -> ()
}

hw.module @test() -> () {
  hw.instance "owo1" @owo() : () -> ()
  hw.instance "nya1" @nya() : () -> ()
}

// RUN: querytool %s --filter "uwu" | FileCheck --check-prefix=TEST2 %s
// TEST2: ::uwu
