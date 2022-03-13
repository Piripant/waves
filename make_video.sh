ffmpeg -r 60 -i images/output%d.png -c:v libx264 -r 60 -pix_fmt yuv420p movie.mp4
