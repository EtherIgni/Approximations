/home/aaron/.local/Sammy/bin/sammy<<EOF
sammy.inp
SAMMY.PAR
sammy.dat


EOF
chi2_line=$(grep -i "CUSTOMARY CHI SQUARED = " SAMMY.LPT | tail -n 1)
chi2_string=$(echo "$chi2_line" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')
chi2_linen=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" SAMMY.LPT | tail -n 1)
chi2_stringn=$(echo "$chi2_linen" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')
echo "$chi2_string $chi2_stringn"
            