#set datafile separator ","
set terminal gif animate delay 5 optimize size 640,480
set output './data_image/HOGE.gif'
set tics font "Arial,10"
set title font "Arial,15"
set xlabel font "Arial,15"
set ylabel font "Arial,15"
set y2label font "Arial,15"
set zlabel font "Arial,15"
set yrange[-3:4]
do for [i=1:350:1] {

filename = sprintf("./data_renban2/rc_out.%04d", i) # n番目のデータファイルの名前の生成
filename2 = sprintf("./data_renban2/rc_out.%04d", i) # n番目のデータファイルの名前の生成
time = sprintf("t=%d[epoch]", i)
set title time
plot filename using 1 title "output(RNN)" with lines lc 15 lw 4,filename2 using 3 title "original" with lines lc 6 lw 2,filename2 using 5 title "input" with lines lt 0 lc 8 lw 0.2

#
}
unset output