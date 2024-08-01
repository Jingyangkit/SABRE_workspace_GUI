import os

cwd = os.getcwd()

os.system("python aquire_best_spectrum.py")
os.system("python 1shot_ensemble.py --meta fc --verbose 1")
os.system("python 1shot_ensemble.py --meta none")
os.system("python 1shot_ensemble.py --meta linear")
os.system("python 1shot_ensemble.py --meta average")
os.system("python 1shot_ensemble.py --meta none_tuned")