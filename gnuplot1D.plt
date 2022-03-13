stats 'out.dat' u 1:2 nooutput
blocks = STATS_blocks

set term png
set xrange [0:2]
set yrange [-1:1]
do for[in=0:blocks-1] {
    set output 'images/output'.in.'.png'
    plot 'out.dat' i in using 1:2 w lines t columnheader(1)
}