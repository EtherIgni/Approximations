##################################
# Generate YW for trans_model
/home/aaron/.local/Sammy/bin/sammy<<EOF
trans_model_iter.inp
iterate/bayes_iter$1.par
trans_model.dat
iterate/bayes_iter$1.cov


EOF
mv -f SAMMY.LPT "iterate/trans_model_iter$1.lpt" 
mv -f SAMMY.ODF "iterate/trans_model_iter$1.odf" 
mv -f SAMMY.LST "iterate/trans_model_iter$1.lst" 
mv -f SAMMY.YWY "iterate/trans_model_iter$1.ywy" 
##################################
# Generate YW for capture_model
/home/aaron/.local/Sammy/bin/sammy<<EOF
capture_model_iter.inp
iterate/bayes_iter$1.par
capture_model.dat
iterate/bayes_iter$1.cov


EOF
mv -f SAMMY.LPT "iterate/capture_model_iter$1.lpt" 
mv -f SAMMY.ODF "iterate/capture_model_iter$1.odf" 
mv -f SAMMY.LST "iterate/capture_model_iter$1.lst" 
mv -f SAMMY.YWY "iterate/capture_model_iter$1.ywy" 
################# read chi2 #######################
#
chi2_line_trans_model=$(grep -i "CUSTOMARY CHI SQUARED =" iterate/trans_model_iter$1.lpt)
chi2_string_trans_model=$(echo "$chi2_line_trans_model" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')
ndat_line_trans_model=$(grep -i "Number of experimental data points = " iterate/trans_model_iter$1.lpt)
ndat_string_trans_model=$(echo "$ndat_line_trans_model" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')

chi2_line_capture_model=$(grep -i "CUSTOMARY CHI SQUARED =" iterate/capture_model_iter$1.lpt)
chi2_string_capture_model=$(echo "$chi2_line_capture_model" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')
ndat_line_capture_model=$(grep -i "Number of experimental data points = " iterate/capture_model_iter$1.lpt)
ndat_string_capture_model=$(echo "$ndat_line_capture_model" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')


echo "$1 $chi2_string_trans_model $chi2_string_capture_model"

echo "$1 $ndat_string_trans_model $ndat_string_capture_model"
