import numpy as np
import os
import concurrent.futures


def load_file(file_path, idx):
    namespace = {}
    contents = open(file_path, 'r').read()
    exec(contents, namespace)
    return np.array(namespace['DGNu']), np.array(namespace['DGNv']), np.array(namespace['DGNp'])


def read_from_dir(directory):
    DIR = 'fine_re200'
    number_of_files = len(os.listdir(DIR))
    TimeStep_DGNu = [None] * number_of_files
    TimeStep_DGNv = [None] * number_of_files
    TimeStep_DGNp = [None] * number_of_files
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}
        for i, file_name in enumerate(sorted(os.listdir(DIR))):
            print(f'Submited for reading {file_name}...', end='\r')
            file_path = os.path.join(DIR, file_name)
            future = executor.submit(load_file, file_path, i)
            futures[future] = i
        print()
        complete = 0
        for future in concurrent.futures.as_completed(futures):
            print(f'Read {complete}/{number_of_files}', end='\r')
            complete += 1
            u, v, p = future.result()
            TimeStep_DGNu[futures[future]] = u
            TimeStep_DGNv[futures[future]] = v
            TimeStep_DGNp[futures[future]] = p
        print()
        return np.array(TimeStep_DGNu), np.array(TimeStep_DGNv), np.array(TimeStep_DGNp), number_of_files
