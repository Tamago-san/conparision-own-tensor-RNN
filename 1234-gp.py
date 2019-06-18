#set terminal x11
#set grid
set datafile separator ","
set tics font "Arial,10"
set title font "Arial,15"
set xlabel font "Arial,15"
set ylabel font "Arial,15"
set y2label font "Arial,15"
set zlabel font "Arial,15"
#set ylabel "y"
set xlabel "epoch"
set title "RNN"
#set ylabel "LYAPNOV"
set ylabel "ERROR"
#set y2tics
set logscal y
set ytics nomirror
set xrange[0:1000]
#set yrange[:1]
rc_err=0.024269096781920487


#! データをプロットする。
#! 「axis x1y1」＝左側のY軸を使用して描画
#! 「axis x1y2」＝右側のY軸を使用して描画
plot "./data_out/lyapnov_end_trstep.1000" using 1:3 axis x1y1  with l  lw 2 title "error tr"
replot "./data_out/lyapnov_end_trstep.1000" using 1:4 axis x1y1  with l  lw 2 title "error hanka"
