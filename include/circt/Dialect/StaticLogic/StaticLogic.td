//===- StaticLogic.td - StaticLogic Definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// StaticLogic Ops are defined in tablegen.
//
//===----------------------------------------------------------------------===//

#ifdef STATICLOGIC_OPS
#else
#define STATICLOGIC_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def StaticLogic_Dialect : Dialect {
  let name = "staticlogic";
  let cppNamespace = "::circt::staticlogic";
}

def PipelineOp : Op<StaticLogic_Dialect, "pipeline", [NoSideEffect]> {
  let summary = "pipeline operation";
  let description = [{
    The "staticlogic.pipeline" operation represents a statically scheduled
    pipeline stucture which contains several MLIR blocks. Each MLIR block is
    corresponding to a pipeline stage.
  }];

  let arguments = (ins Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
  let regions = (region AnyRegion: $body);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins "ValueRange":$operands, "ValueRange":$results), [{
    SmallVector<Type, 4> argTypes;
    for (auto value : operands)
      argTypes.push_back(value.getType());

    SmallVector<Type, 4> resultTypes;
    for (auto value : results)
      resultTypes.push_back(value.getType());

    Region *bodyRegion = odsState.addRegion();
    Block *body = new Block();
    bodyRegion->push_back(body);
    body->addArguments(argTypes);

    odsState.addOperands(operands);
    odsState.addTypes(resultTypes);
  }]>];
}

def ReturnOp : Op<StaticLogic_Dialect, "return", [Terminator]> {
  let summary = "StaticLogic dialect return.";
  let description = [{
    The "staticlogic.return" operation represents a terminator of a statically
    scheduled module, which is similar to a standard return operation.
  }];

  let arguments = (ins Variadic<AnyType>: $operands);
}

def PipelineWhileOp : Op<StaticLogic_Dialect, "pipeline.while", []> {
  let summary = "StaticLogic dialect pipeline while-loop.";
  let description = [{
    The `staticlogic.pipeline.while` operation represents a statically scheduled
    pipeline stucture that executes while a condition is true. For more details,
    see: https://llvm.discourse.group/t/rfc-representing-pipelined-loops/4171.

    A pipeline captures the result of scheduling, and is not generally safe to
    transform, besides lowering to hardware dialects. For more discussion about
    relaxing this, see: https://github.com/llvm/circt/issues/2204.

    This is the top-level operation representing a high-level pipeline. It is
    not isolated from above, but could be if this is helpful. A pipeline
    contains two regions: `condition` and `stages`.

    The pipeline may accept an optional `iter_args`, similar to the SCF dialect,
    for representing loop-carried values like induction variables or reductions.
    When the pipeline starts execution, the registers indicated as `iter_args`
    by `staticlogic.pipeline.terminator` should be initialized to the initial
    values specified in the `iter_args` section here. The `iter_args` relate to
    the initiation interval of the loop. The maximum distance in stages between
    where an `iter_arg` is used and where that `iter_arg` is registered must be
    less than the loop's initiation interval. For example, with II=1, each
    `iter_arg` must be used and registered in the same stage.

    The single-block `condition` region dictates the condition under which the
    pipeline should execute. It has a `staticlogic.register` terminator, and the
    pipeline initiates new iterations while the registered value is `true : i1`.
    It may access SSA values dominating the pipeline, as well as `iter_args`,
    which are block arguments. The body of the block may only contain
    "combinational" operations, which are currently defined to be simple
    arithmetic, comparisons, and selects from the `Standard` dialect.

    The single-block `stages` region wraps `staticlogic.pipeline.stage`
    operations. It has a `staticlogic.pipeline.terminator` terminator, which can
    both return results from the pipeline and register `iter_args`. Stages may
    access SSA values dominating the pipeline, as well as `iter_args`, which are
    block arguments.
  }];

  let arguments = (ins
    I64Attr:$II,
    Variadic<AnyType>:$iterArgs
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region
    SizedRegion<1>:$condition,
    SizedRegion<1>:$stages
  );

  let parser = [{
    return parse$cppClass(parser, result);
  }];

  let printer = [{
    print$cppClass(p, *this);
  }];

  let verifier = [{
    return ::verify$cppClass(*this);
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::IntegerAttr":$II,
                   "mlir::ValueRange":$iterArgs)>
  ];

  let extraClassDeclaration = [{
    Block &getCondBlock() { return condition().front(); }
    Block &getStagesBlock() { return stages().front(); }
  }];
}

def PipelineStageOp : Op<StaticLogic_Dialect, "pipeline.stage",
    [HasParent<"PipelineWhileOp">]> {
  let summary = "StaticLogic dialect pipeline stage.";
  let description = [{
    This operation has a single-block region which dictates the operations that
    may occur concurrently.

    It has a `start` attribute, which indicates the start cycle for this stage.

    It may have an optional `when` predicate, which supports conditional
    execution for each stage. This is in addition to the `condition` region that
    controls the execution of the whole pipeline. A stage with a `when`
    predicate should only execute when the predicate is `true : i1`, and push a
    bubble through the pipeline otherwise.

    It has a `staticlogic.register` terminator, which passes the concurrently
    computed values forward to the next stage.

    Any stage may access `iter_args`. If a stage accesses an `iter_arg` after
    the stage in which it is defined, it is up to lowering passes to preserve
    this value until the last stage that needs it.

    Other than `iter_args`, stages may only access SSA values dominating the
    pipeline or SSA values computed by a previous stage. This ensures the stages
    capture the coarse-grained schedule of the pipeline and how values feed
    forward and backward.
  }];

  let arguments = (ins
    SI64Attr:$start,
    Optional<I1>:$when
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = [{
    `start` `=` $start (`when` $when^)? $body (`:` qualified(type($results))^)? attr-dict
  }];

  let verifier = [{
    return ::verify$cppClass(*this);
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::IntegerAttr":$start)>
  ];

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return body().front(); }
    unsigned getStageNumber();
  }];
}

def PipelineRegisterOp : Op<StaticLogic_Dialect, "pipeline.register",
    [ParentOneOf<["PipelineWhileOp", "PipelineStageOp"]>, Terminator]> {
  let summary = "StaticLogic dialect pipeline register.";
  let description = [{
    The `staticlogic.pipeline.register` terminates a pipeline stage and
    "registers" the values specified as operands. These values become the
    results of the stage.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let assemblyFormat = [{
    $operands (`:` qualified(type($operands))^)? attr-dict
  }];

  let verifier = [{
    return ::verify$cppClass(*this);
  }];
}

def PipelineTerminatorOp : Op<StaticLogic_Dialect, "pipeline.terminator",
    [HasParent<"PipelineWhileOp">, Terminator, AttrSizedOperandSegments]> {
  let summary = "StaticLogic dialect pipeline terminator.";
  let description = [{
    The `staticlogic.pipeline.terminator` operation represents the terminator of
    a `staticlogic.pipeline.while`.

    The `results` section accepts a variadic list of values which become the
    pipeline’s return values. These must be results of a stage, and their types
    must match the pipeline's return types. The results need not be defined in
    the final stage, and it is up to lowering passes to preserve these values
    until the final stage is complete.

    The `iter_args` section accepts a variadic list of values which become the
    next iteration’s `iter_args`. These may be the results of any stage, and
    their types must match the pipeline's `iter_args` types.
  }];

  let arguments = (ins
    Variadic<AnyType>:$iter_args,
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    `iter_args` `(` $iter_args `)` `,`
    `results` `(` $results `)` `:`
    functional-type($iter_args, $results) attr-dict
  }];

  let verifier = [{
    return ::verify$cppClass(*this);
  }];
}

#endif // STATICLOGIC_OPS
