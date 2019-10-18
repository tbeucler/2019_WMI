#!/bin/bash

SL='SL_SWD_'
FIL='SWD_'
MEM=10000
SCRIPT="012_SWD"
YEAR=(2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018)
echo 'SL='$SL' FIL='$FIL' MEM='$MEM' SCRIPT='$SCRIPT' YEAR='$YEAR
for ind in {0..18}; do
  i=$((${YEAR[$ind]}))
  echo $i
  sed '2 s/[0-9][0-9][0-9][0-9]/'$i'/' $SCRIPT".py" >$SCRIPT$i".py"
  ./gen_slurm_script 1 sched_mit_twcronin $SL$i".out" $SL$i".err" $MEM tbeucler@uci.edu $SCRIPT$i".py" >> $FIL$i".slurm" 
  sbatch $FIL$i".slurm"
  rm $FIL$i".slurm"
done
