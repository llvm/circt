(set-logic HORN)
(declare-rel A (Int Int))
(declare-rel B (Int Int))
(declare-rel C (Int Int))
(declare-rel CB (Int))
(declare-var input2 Int)
(declare-var input Int)
(declare-var time Int)

(rule (=> (= time 0) (A input time)))

(rule (=> (and (A input time) (distinct input 1)) (B input2 (+ time 1))))

(rule (=> (and (B input time) (= input 1)) (C input2 (+ time 1))))

(rule (=> (and (B input time) (= input 1)) (A input2 (+ time 1))))

(rule (=> (and (B input time) (distinct input 3)  ) (B input2 (+ time 1))))

(rule (=> (and (C input time) (= input 1)  ) (A input2 (+ time 1))))

(rule (=> (and (C input time) (distinct input 1)  ) (C input2 (+ time 1))))

(query A :print-certificate true)