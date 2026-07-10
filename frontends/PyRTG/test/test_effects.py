# RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s

from pyrtg import (test, config, Config, effect, perform, effect_handler,
                   Integer, IntegerType)


# A void effect: takes an index, resumes with nothing.
# CHECK-LABEL: rtg.effect @log_value : (index) -> ()
@effect(inputs=[IntegerType()])
def log_value(n):
  ...


# A value effect: takes an index, resumes with an index.
# CHECK-LABEL: rtg.effect @choose : (index) -> index
@effect(inputs=[IntegerType()], result=IntegerType())
def choose(n):
  ...


@config
class Singleton(Config):
  pass


# A simple handler for a void effect auto-resumes with no value.
# CHECK-LABEL: rtg.test @test_void
# CHECK: rtg.with_handlers {
# CHECK:   handle @log_value(%{{.+}}: index{{.*}}: !rtg.continuation<none>
# CHECK:     rtg.resume %{{.+}} : <none>
# CHECK:   do {
# CHECK:     rtg.perform @log_value(%{{.+}}) : (index) -> none
@test(Singleton)
def test_void(config):

  def handler(n):
    # Simple void handler: framework auto-resumes.
    pass

  with effect_handler(log_value, handler):
    perform(log_value, Integer(42))


# A simple handler for a value effect auto-resumes with its return value.
# CHECK-LABEL: rtg.test @test_value_simple
# CHECK: rtg.with_handlers {
# CHECK:   handle @choose(%{{.+}}: index{{.*}}: !rtg.continuation<index>
# CHECK:     rtg.resume %{{.+}}, %{{.+}} : <index>, index
# CHECK:   do {
# CHECK:     %{{.+}} = rtg.perform @choose(%{{.+}}) : (index) -> index
@test(Singleton)
def test_value_simple(config):

  def handler(n):
    # Simple value handler: return value is used to resume.
    return n

  with effect_handler(choose, handler):
    perform(choose, Integer(7))


# A control handler receives the continuation and resumes it explicitly.
# CHECK-LABEL: rtg.test @test_value_control
# CHECK: rtg.with_handlers {
# CHECK:   handle @choose(%{{.+}}: index{{.*}}: !rtg.continuation<index>
# CHECK:     rtg.resume %{{.+}}, %{{.+}} : <index>, index
# CHECK:   do {
# CHECK:     %{{.+}} = rtg.perform @choose(%{{.+}}) : (index) -> index
@test(Singleton)
def test_value_control(config):

  def handler(n, k):
    # Control handler: arity is len(inputs) + 1, resumes explicitly.
    k.resume(n)

  with effect_handler(choose, handler):
    perform(choose, Integer(3))


# A single scope can install handlers for several effects at once.
# CHECK-LABEL: rtg.test @test_multi
# CHECK: rtg.with_handlers {
# CHECK:   handle @log_value(%{{.+}}: index{{.*}}: !rtg.continuation<none>
# CHECK:   handle @choose(%{{.+}}: index{{.*}}: !rtg.continuation<index>
# CHECK:   do {
# CHECK:     rtg.perform @log_value(%{{.+}}) : (index) -> none
# CHECK:     %{{.+}} = rtg.perform @choose(%{{.+}}) : (index) -> index
@test(Singleton)
def test_multi(config):

  def log_h(n):
    pass

  def choose_h(n):
    return n

  with effect_handler([(log_value, log_h), (choose, choose_h)]):
    perform(log_value, Integer(1))
    perform(choose, Integer(2))
