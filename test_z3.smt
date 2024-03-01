(set-info :status unknown)
(declare-fun A (Int Int) Bool)
(declare-fun B (Int Int) Bool)
(assert
 (A 0 0)
 )
(assert
 (forall ((time Int) )(forall((arg0 Int))(let (($x39 (A arg0 time)))
 (=> (and (and (>= time 0) (< time 10)) $x39) (B arg0 (+ time 1))))))
 )
(assert
 (forall ((time Int) )(forall((arg0 Int))(let (($x51 (B arg0 time)))
 (let (($x52 (and $x51 (= arg0 5))))
  (=> (and (>= time 0) (< time 10))  (=> $x52 (A 0 (+ time 1))))))))
 )
(assert
 (forall ((time Int) )(forall((arg0 Int))(let (($x51 (and (B arg0 time)(distinct arg0 5))))
  (=> (and (>= time 0) (< time 10))   (=> $x51 (B (+ arg0 1) (+ time 1)))))))
 )
(assert
 (forall ((time Int) )(forall((arg0 Int))
  (=> (and (>= time 0) (< time 10))   (=> (and (B arg0 time) (> arg0 5)) false)))
 ))
(assert
 (forall ((time Int) ) (forall((arg0 Int))
  (=> (and (>= time 0) (< time 10))  (=> (and (> arg0 0) (A arg0 time)) false)))
 )
)
(check-sat)
(get-model)