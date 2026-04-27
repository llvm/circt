// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Verifies Moore class attributes populated from slang:
//   * ClassMethodDeclOp.is_virtual from MethodFlags::Virtual
//   * ClassMethodDeclOp.is_override from SubroutineSymbol::getOverride()
//   * ClassDeclOp.is_virtual from ClassType::isAbstract

class BaseComponent;
  virtual function void build_phase();
  endfunction
endclass

class DerivedComponent extends BaseComponent;
  function void build_phase();
  endfunction
endclass

virtual class AbstractComponent;
  pure virtual function void abstract_method();
endclass

// CHECK-LABEL: moore.class.classdecl @BaseComponent
// CHECK: moore.class.methoddecl @build_phase
// CHECK-SAME: is_virtual

// CHECK-LABEL: moore.class.classdecl @DerivedComponent
// CHECK: moore.class.methoddecl @build_phase
// CHECK-SAME: is_override

// CHECK-LABEL: moore.class.classdecl @AbstractComponent
// CHECK-SAME: is_virtual
// CHECK: moore.class.methoddecl @abstract_method
// CHECK-SAME: is_virtual
