set termoption font "Helvetica,15"
set yrange [3.2:4.5]
plot 'ConcPot' u ($2/$3/28000.0):($7/$8) w l lw 3 title "EchemAMR",\
     '1c_p2d' u 1:2 w lp lw 1 ps 2 pt 6 pi 10 title "P2D",\
     '1c_sphere_fenics' u 1:2 w lp lw 1 ps 2 pt 8 title "Fenics",\
      
