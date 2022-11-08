"""
    file_utils.py

    Summary:
        Contains general non-math related utility functions.

    Author: Trym Tengesdal
"""


from pathlib import Path

import yaml


def read_yaml_into_dict(file_name: Path) -> dict:
    with file_name.open(mode="r", encoding="utf-8") as file:
        output_dict = yaml.safe_load(file)
    return output_dict


# necessary??
# def move_xlsx_files():
#     for file in os.listdir():
#         if file.endswith(".xlsx"):
#             if len(file) == 17:
#                 shutil.move(file, "output/eval/ship" + str(file[11]) + "/" + file)
#             if len(file) == 18:
#                 shutil.move(file, "output/eval/ship" + str(file[11]) + str(file[12]) + "/" + file)
#             if len(file) == 19:
#                 shutil.move(file, "output/eval/ship" + str(file[11]) + str(file[12]) + str(file[13]) + "/" + file)
