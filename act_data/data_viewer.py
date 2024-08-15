from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', help = "the name of the task")
parser.add_argument('--episode', help = "the episode number")
args = parser.parse_args()
task_name = args.task_name
episode = args.episode

import h5py
file_path = Path(".") / "demonstrations" / "hdf5" / task_name / f"episode_{episode}.hdf5"
f = h5py.File(file_path, 'r')
print(f.keys())
print(f"attr={dict(f.attrs)}")

def printname(name):
    print(name)
    if isinstance(f[name], h5py.Dataset):
        print(f[name].shape, f[name].dtype)

print(len(f['actions/eef_pose'][()]))
        
f.visit(printname)