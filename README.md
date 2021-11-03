# __GUIDE TO USE THIS CODE__

## __TABLE OF CONTENTS__
- [__GUIDE TO USE THIS CODE__](#guide-to-use-this-code)
  - [__TABLE OF CONTENTS__](#table-of-contents)
  - [__REQUIREMENTS__](#requirements)
  - [__HOW TO USE__](#how-to-use)
  - [__ADDITIONAL INFORMATION__](#additional-information)

## __REQUIREMENTS__
- python >= 3.7
- torch
- numpy
- pandas
- xlsxwriter and openpyxl (to save the result and generated dataset in .xlsx format)

## __HOW TO USE__
There are some main program (the file name starts with `main`) where each file has a slight different approach to detect symbols. But, those files were built based on DIP algorithm. The detailed approach is described in the following file. You can run those files to get the results.

The results will be stored at directory `data`. The matrices will be stored in `.xlsx` format and SER calculations will be saved in `.txt` file.

## __ADDITIONAL INFORMATION__
- Model code is stored in directory `models` -> only for main_old. The other main program doesn't use functions in model directory
- The model in main_old program didn't do downsampling and upsampling. If you want to modify it, please modify `skip.py` in directory `models`
- This github is still being developed
- Feel free to modify the code to get more informations