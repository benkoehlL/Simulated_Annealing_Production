set border lw 2.0
# margins 
set tmargin 1.5
set bmargin 3.5
set lmargin 10.5
set rmargin 3.5

set xlabel "log(1/T)" font ",18" 
set ylabel "total tardiness/10^6" font ",18" offset -1,0
set xtics font ",14"
set ytics font ",14"
set label '.' at graph -10.19,-0.91 textcolor rgb "white"
set xrange[1:10.0]
set yrange[-0.1:1.2]
plot 'tardiness.dat' u ((log($1)+10)/10):($2/7.72141e+06) w l lw 3 lc rgb "dark-blue" t ''
