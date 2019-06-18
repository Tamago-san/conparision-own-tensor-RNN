#set terminal x11
#set grid
set datafile separator ","
set tics font "Arial,10"
set title font "Arial,15"
set xlabel font "Arial,15"
set ylabel font "Arial,15"
#set y2label font "Arial,15"
set zlabel font "Arial,15"
#set ylabel "y"
set xlabel "STEP"
set title "RNN-OWN"
set ylabel ""
set xrange[0:1000]
#set y2range [0:2.2]
#set yrange [-3.70:-3.67]

plot "./data_renban2/rc_out.0725" using 1 title "output(RNN)" with lines lc 15 lw 4
replot "./data_renban2/rc_out.0725" using 2 title "original" with lines lc 6 lw 2
