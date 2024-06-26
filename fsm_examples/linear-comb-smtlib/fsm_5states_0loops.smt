; 
(set-info :status unknown)
(declare-sort STATE 0)
 (declare-fun _0 () STATE)
(declare-fun time-to-state (Int) STATE)
(declare-fun var0_0 (Int) Int)
(declare-fun _1 () STATE)
(declare-fun _2 () STATE)
(declare-fun _3 () STATE)
(declare-fun _4 () STATE)
(declare-fun _5 () STATE)
(assert
 (let ((?x13 (time-to-state 0)))
 (= ?x13 _0)))
(assert
 (let ((?x33 (var0_0 0)))
 (= ?x33 0)))
(assert
 (let ((?x33 (var0_0 0)))
 (= ?x33 0)))
(assert
 (forall ((time Int) )(let (($x52 (= (var0_0 (+ time 1)) (+ (var0_0 time) 1))))
 (let ((?x56 (time-to-state time)))
 (let (($x57 (= ?x56 _0)))
 (=> $x57 (and (= (time-to-state (+ time 1)) _1) $x52))))))
 )
(assert
 (forall ((time Int) )(let (($x52 (= (var0_0 (+ time 1)) (+ (var0_0 time) 1))))
 (let ((?x56 (time-to-state time)))
 (let (($x74 (= ?x56 _1)))
 (=> $x74 (and (= (time-to-state (+ time 1)) _2) $x52))))))
 )
(assert
 (forall ((time Int) )(let (($x52 (= (var0_0 (+ time 1)) (+ (var0_0 time) 1))))
 (let ((?x56 (time-to-state time)))
 (let (($x84 (= ?x56 _2)))
 (=> $x84 (and (= (time-to-state (+ time 1)) _3) $x52))))))
 )
(assert
 (forall ((time Int) )(let (($x52 (= (var0_0 (+ time 1)) (+ (var0_0 time) 1))))
 (let ((?x56 (time-to-state time)))
 (let (($x94 (= ?x56 _3)))
 (=> $x94 (and (= (time-to-state (+ time 1)) _4) $x52))))))
 )
(assert
 (forall ((time Int) )(let (($x52 (= (var0_0 (+ time 1)) (+ (var0_0 time) 1))))
 (let ((?x56 (time-to-state time)))
 (let (($x104 (= ?x56 _4)))
 (=> $x104 (and (= (time-to-state (+ time 1)) _5) $x52))))))
 )
(assert
 (forall ((time Int) )(let ((?x56 (time-to-state time)))
 (let (($x104 (= ?x56 _4)))
 (= $x104 (and (distinct (var0_0 time) 4) true)))))
 )
(assert
 (and (distinct _0 _1 _2 _3 _4 _5) true))
(check-sat)
