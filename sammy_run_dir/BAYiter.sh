#######################################################
# Run Bayes for bayes_iter$plus_one
#
#######################################################
/home/aaron/.local/Sammy/bin/sammy<<eod
solvebayes_iter.inp
iterate/bayes_iter$1.par
dummy.dat
iterate/trans_model_iter$1.ywy
iterate/capture_model_iter$1.ywy

iterate/bayes_iter$1.cov

eod
plus_one=$(( $1 + 1 ))
mv -f SAMMY.LPT iterate/bayes_iter$plus_one.lpt 
mv -f SAMMY.PAR iterate/bayes_iter$plus_one.par 
mv -f SAMMY.COV iterate/bayes_iter$plus_one.cov 
rm -f SAM*
