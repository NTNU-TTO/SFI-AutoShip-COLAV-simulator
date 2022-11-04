"""Contains general non-math related utility functions."""

import os
import shutil


# necessary??
def move_xlsx_files():
    for file in os.listdir():
        if file.endswith(".xlsx"):
            if len(file) == 17:
                shutil.move(file, "output/eval/ship" + str(file[11]) + "/" + file)
            if len(file) == 18:
                shutil.move(file, "output/eval/ship" + str(file[11]) + str(file[12]) + "/" + file)
            if len(file) == 19:
                shutil.move(file, "output/eval/ship" + str(file[11]) + str(file[12]) + str(file[13]) + "/" + file)
