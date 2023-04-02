import numpy as np
import csv


# with open("x_data.npy", 'rb') as f_read, open("x_data_new.npy", 'wb') as f_write:
#     lines = f_read.readlines()
#     new_lines = []
#     for idx, line in enumerate(lines):
#         new_lines.append(line)
#         if b"NUMPY" in line and idx != 0:
#             for c, char in enumerate(line):
#                 if char == b'N':
#                     line = line.insert(c-2, '\n')
#             new_lines.insert(len(new_lines) - 2, '\n')
#     f_write.writelines(new_lines)

with open("x_data.npy", 'rb') as f:
    lines = f.readlines()
    np_bof = [idx for idx, line in enumerate(lines) if b'NUMPY' in line]
    for pos in np_bof:
        f.seek(pos, 0)
        x_load = np.load(f, allow_pickle=True)
        for i in range(x_load.shape[0]):
            print("pt: " + str(x_load[i].item().get('pt')))
            print("dd: " + str(x_load[i].item().get('dd')))


y_load = np.load("y_data.npy", allow_pickle=True)
for i in range(y_load.shape[0]):
    for m in y_load[i][0][0].keys():
        print("m" + str(m) + ": " + str(y_load[i][0][0].get(m)))

