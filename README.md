# AlphaGomoku
> Gomoku Game with AI

![](./pics/gomoku.jpeg)

## Features
+ Reinforcement learning

## Dependencies
+ Python 3+
+ Numpy
+ H5py
+ PyQt5

## Status
Under development. Not all codes can run normally. May with some obscure bugs

## How to use

### Play with AI based on complex rules
+ Install necessary dependencies (Linux)  
    ```pip install numpy h5py PyQt5```
+ Play with naive computer (Linux)  
    ```$ cd gomoku/test```  
    ```$ python playwithnaiveai.py```

### Train deep learning AI
+ Re-Train AI network (Linux)  
    ```$ cd alphagomoku```  
    ```$ python Alphagomoku.py --retrain --verbose```

### Play with deep learning AI
Please train AI network firstly!

+ Play with AI network (Linux)  
    ```$ cd alphagomoku```  
    ```$ python Alphagomoku.py --play```