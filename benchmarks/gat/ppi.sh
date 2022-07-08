#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=10] span[ptile=1]"
#BSUB -W 0:59
#BSUB -n 1
#BSUB -gpu "num=1/task:j_exclusive=yes:mode=shared"

python ppi.py


