source ~/harienv.sh
srun -n 64 python avg_data_cathode.py "plt?????" "Potential"
srun -n 64 python avg_data_anode.py "plt?????" "Concentration"
paste Concentrationavg Potentialavg > ConcPot
