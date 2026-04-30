##################################
# Plot for trans_model
/home/aaron/.local/Sammy/bin/sammy<<EOF
trans_model_plot.inp
results/step$1.par
trans_model.dat


EOF
mv -f SAMMY.LPT "results/trans_model.lpt" 
mv -f SAMMY.ODF "results/trans_model.odf" 
mv -f SAMMY.LST "results/trans_model.lst" 

##################################
# Plot for capture_model
/home/aaron/.local/Sammy/bin/sammy<<EOF
capture_model_plot.inp
results/step$1.par
capture_model.dat


EOF
mv -f SAMMY.LPT "results/capture_model.lpt" 
mv -f SAMMY.ODF "results/capture_model.odf" 
mv -f SAMMY.LST "results/capture_model.lst" 

################# read chi2 #######################
#
chi2_line_trans_model=$(grep -i "CUSTOMARY CHI SQUARED =" results/trans_model.lpt | tail -n 1)
chi2_string_trans_model=$(echo "$chi2_line_trans_model" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')

chi2n_line_trans_model=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" results/trans_model.lpt | tail -n 1)
chi2n_string_trans_model=$(echo "$chi2n_line_trans_model" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')

chi2_line_capture_model=$(grep -i "CUSTOMARY CHI SQUARED =" results/capture_model.lpt | tail -n 1)
chi2_string_capture_model=$(echo "$chi2_line_capture_model" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')

chi2n_line_capture_model=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" results/capture_model.lpt | tail -n 1)
chi2n_string_capture_model=$(echo "$chi2n_line_capture_model" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')


echo "$1 $chi2_string_trans_model $chi2_string_capture_model $chi2n_string_trans_model $chi2n_string_capture_model"
