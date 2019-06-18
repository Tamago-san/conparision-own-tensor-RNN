#set terminal x11
#set grid
set tics font "Arial,10"
set title font "Arial,15"
set xlabel font "Arial,15"
set ylabel font "Arial,15"
set y2label font "Arial,15"
set zlabel font "Arial,15"
#set ylabel "y"
#set xlabel "epoch"
set title "RNN"
#set ylabel "LYAPNOV"
set ylabel "ERROR"
#set y2tics
#set logscal y2
set ytics nomirror
set xrange[0:1000]
#set xrange[20:35]
#set yrange[0:0.6]
rc_err=0.07140703008014705
rc_lyapnov=-0.656803

#! 右側のY軸を使用可能にする
#set y2tics
#! 右側のY軸の範囲を設定する
#set y2range [0:2.2]
#set yrange [-3.70:-3.67]

#plot "./data_out/lyapnov_end_01.dat" using 1:2 axis x1y1  with linespoints pt 7 lc 2 lw 2 title "RNN-LYAPNOV"
#replot rc_lyapnov axis x1y1  with lines dt (5,5) lc 2 lw 2 title "RC -LYAPNOV"
#replot "./data_out/lyapnov_end_01.dat" using 1:3 axis x1y2  with linespoints pt 7 lc 15 lw 2 title "RNN-ERROR"
#replot rc_err axis x1y2  with lines dt (5,5) lc 15 lw 2 title "RC -ERROR"


#plot "./data_out/lyapnov_end_trstep.0010" using 1:3  with linespoints pt 7 lc 1 lw 2 title "10"
plot "./data_out/lyapnov_end_trstep.0100" using 1:5 with line pt 7  lw 2.5 title "100"
replot "./data_out/lyapnov_end_trstep.0500" using 1:5 with line pt 7  lw 2.5 title "500"
#replot "./data_out/lyapnov_end_trstep.0900" using 1:5 with line pt 7  lw 2.5 title "900"
replot "./data_out/lyapnov_end_trstep.1000" using 1:5 with line pt 7  lw 2.5 title "1000"
replot "./data_out/lyapnov_end_trstep.1400" using 1:5 with line pt 7  lw 2.5 title "1400"
replot "./data_out/lyapnov_end_trstep.1500" using 1:5 with line pt 7  lw 2.5 title "1500"
#replot "./data_out/lyapnov_end_trstep.2.5000" using 1:3 with line pt 7  lw 2.5 title "2000"
replot "./data_out/lyapnov_end_trstep.1600" using 1:5 with line pt 7  lw 2.5 title "1600"
#replot "./data_out/lyapnov_end_trstep.1500" using 1:3 with linespoints pt 7  lw 2 title "1500"
#replot "./data_out/lyapnov_end_trstep.1200" using 1:3 with linespoints pt 7  lw 2 title "1200"
#replot "./data_out/lyapnov_end_trstep.1300" using 1:3 with linespoints pt 7  lw 2 title "1300"
#plot "./data_out/lyapnov_end_trstep.1500" using 1:3 with linespoints pt 7  lw 2 title "1500"

#replot "./data_out/lyapnov_end_trstep.0800" using 1:3 with linespoints pt 7  lw 2 title "800"
#replot "./data_out/lyapnov_end_trstep.0200" using 1:3 with linespoints pt 7 lc 9 lw 2 title "200"
#replot "./data_out/lyapnov_end_trstep.0011" using 1:3 axis x1y2 with linespoints pt 7 lc 10 lw 2 title "0011"
#replot "./data_out/lyapnov_end_trstep.0012" using 1:3 axis x1y2 with linespoints pt 7 lc 1 lw 2 title "0012"
#replot "./data_out/lyapnov_end_trstep.0013" using 1:3 axis x1y2 with linespoints pt 7 lc 2 lw 2 title "0013"
#replot "./data_out/lyapnov_end_trstep.0014" using 1:3 axis x1y2 with linespoints pt 7 lc 3 lw 2 title "0014"
#replot "./data_out/lyapnov_end_trstep.0015" using 1:3 axis x1y2 with linespoints pt 7 lc 4 lw 2 title "0015"
#replot "./data_out/lyapnov_end_trstep.0016" using 1:3 axis x1y2 with linespoints pt 7 lc 5 lw 2 title "0016"
#replot "./data_out/lyapnov_end_trstep.0017" using 1:3 axis x1y2 with linespoints pt 7 lc 6 lw 2 title "0017"
#replot "./data_out/lyapnov_end_trstep.0018" using 1:3 axis x1y2 with linespoints pt 7 lc 7 lw 2 title "0018"
#replot "./data_out/lyapnov_end_trstep.0019" using 1:3 axis x1y2 with linespoints pt 7 lc 8 lw 2 title "0019"
#replot "./data_out/lyapnov_end_trstep.0100" using 1:3  with linespoints pt 7 lc 6 lw 2 title "50"
#replot "./data_out/lyapnov_end_trstep.1000" using 1:3  with linespoints pt 7 lc 2 lw 2 title "5"
#replot "./data_out/lyapnov_end_trstep.0040" using 1:3 axis x1y2 with linespoints pt 7 lc 3 lw 2 title "400"
#replot "./data_out/lyapnov_end_trstep.0050" using 1:3 axis x1y2 with linespoints pt 7 lc 4 lw 2 title "500"

#replot rc_err axis x1y2  with lines dt (5,5) lc 15 lw 2 title "RC -ERROR"