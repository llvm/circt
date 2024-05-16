// Check textual verification
// RUN: circt-opt %s | FileCheck %s

// Check explicit verification pass
// RUN: circt-opt %s -verify-diagnostics --pass-pipeline="builtin.module(ibis.design(hw-verify-irn))"

// CHECK-LABEL:  ibis.design @x {
// CHECK-NEXT:    ibis.container @Foo {
// CHECK-NEXT:      %this = ibis.this <@Foo> 
// CHECK-NEXT:      ibis.container @Bar {
// CHECK-NEXT:        %this_0 = ibis.this <@Foo::@Bar> 
// CHECK-NEXT:        ibis.container @Baz {
// CHECK-NEXT:          %this_1 = ibis.this <@Foo::@Bar::@Baz> 
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    ibis.container @Top {
// CHECK-NEXT:      %this = ibis.this <@Top> 
// CHECK-NEXT:      %foo = ibis.container.instance @foo, <@Foo> 
// CHECK-NEXT:      %bar = ibis.container.instance @bar, <@Foo::@Bar> 
// CHECK-NEXT:      %baz = ibis.container.instance @baz, <@Foo::@Bar::@Baz> 
// CHECK-NEXT:    }
// CHECK-NEXT:  }

ibis.design @x {
  ibis.container @Foo {
    %this_foo = ibis.this <@Foo>
    ibis.container @Bar {
      %this_bar = ibis.this <@Foo::@Bar>      
      ibis.container @Baz {
        %this_baz = ibis.this <@Foo::@Bar::@Baz>
      }
    }
  }

  ibis.container @Top {
    %this = ibis.this <@Top>
    %foo = ibis.container.instance @foo, <@Foo>
    %bar = ibis.container.instance @bar, <@Foo::@Bar>
    %baz = ibis.container.instance @baz, <@Foo::@Bar::@Baz>
  }
}
