#set terminal x11
set grid
set tics font "Arial,10"
set xlabel font "Arial,15"
set ylabel font "Arial,15"
set zlabel font "Arial,15"
set ylabel "y"
set xlabel "x"
set zlabel "z"
set xrange[0:100]

#plot "./data_renban/lyapnov.0031" using 1:2 with line
#replot "./data_renban/lyapnov.0041" using 1:2 with line
#replot "./data_renban/lyapnov.0050" using 1:2 with line
#plot "./data_renban/lyapnov.0250" using 1:3 with line
#plot "./data_renban/lyapnov.0250" using 1:2 with line

#plot "./data_renban2/rc_out.0003" using 4 with line
plot "./data_renban2/rc_out.0001" using 1 with line
replot "./data_renban2/rc_out.0001" using 2 with line
replot "./data_renban2/rc_out.0001" using 3 with line