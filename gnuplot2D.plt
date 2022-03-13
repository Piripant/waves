# line styles for ColorBrewer RdBu
# for use with divering data
# provides 8 colors with red low, white middle, and blue high
# compatible with gnuplot >=4.2
# author: Anna Schneider

# line styles
set style line 1 lt 1 lc rgb '#B2182B' # red
set style line 2 lt 1 lc rgb '#D6604D' # red-orange
set style line 3 lt 1 lc rgb '#F4A582' # 
set style line 4 lt 1 lc rgb '#FDDBC7' # pale orange
set style line 5 lt 1 lc rgb '#D1E5F0' # pale blue
set style line 6 lt 1 lc rgb '#92C5DE' # 
set style line 7 lt 1 lc rgb '#4393C3' # medium blue
set style line 8 lt 1 lc rgb '#2166AC' # dark blue

# palette
set palette defined ( 0 '#B2182B',\
    	    	      1 '#D6604D',\
		      2 '#F4A582',\
		      3 '#FDDBC7',\
		      4 '#D1E5F0',\
		      5 '#92C5DE',\
		      6 '#4393C3',\
		      7 '#2166AC' )



stats 'out.dat' u 1:2 nooutput
blocks = STATS_blocks

set term png
set xrange [-0.1:2.1]
set yrange [-0.1:2.1]
set cbrange [-1.0:1.0]
do for[in=0:blocks-1] {
    set output 'images/output'.in.'.png'
    # set multiplot layout 1,2
    plot 'out.dat' i in using 1:2:3 w image t columnheader(1)
    # plot 'out.dat' i in using 1:2:4 w image t columnheader(1)
    # unset multiplot
}