#!/bin/bash

for graph in ER BA 
do
    for pnt_v in 1e-8 1e-4
    do 
        for E_v in 2 3 4 5
        do 
            python synth_LIN.py --graphtype $graph --log --pnt $pnt_v --E $E_v
        done 
    done

    for pnt_v in 1e-8 1e-4
    do 
        for E_v in 2 3 4 5
        do 
            python synth_LIN.py --graphtype $graph --log --uneven --pnt $pnt_v --E $E_v
        done 
    done
done 
