#######################################################
# Run Bayes for bayes_iter1
#
#######################################################
/home/aaron/.local/Sammy/bin/sammy<<eod
solvebayes_initial.inp
results/step$1.par
dummy.dat
iterate/trans_model_iter0.ywy
iterate/capture_model_iter0.ywy



eod
mv -f SAMMY.LPT iterate/bayes_iter1.lpt 
mv -f SAMMY.PAR iterate/bayes_iter1.par 
mv -f SAMMY.COV iterate/bayes_iter1.cov 
rm -f SAM*
