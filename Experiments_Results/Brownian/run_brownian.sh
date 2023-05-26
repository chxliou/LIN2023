#!/bin/bash

for graph in ER BA 
do
    for dim_v in 1 2
    do
        for pnt_v in 1e-8 1e-4
        do 
            for E_v in 2 3 4 5
            do 
                python Brownian_LIN.py --graphtype $graph --log --dim $dim_v --pnt $pnt_v --E $E_v
            done 
        done
    done
done 

