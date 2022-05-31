% lecture_DRL_coding2
# 2. Coding Exercise, PPO

**First**: To be able to work on the coding exercise you first have to set up python and an IDE (look into slides Coding1_Introduction)

**Second**: When python is running and you set up an IDE to work with, you have to install all dependencies. Therefore:
(You find more detailed instructions on installing virtual environments and dependencies in tutorials - you find the links to these tutorials in the presentation Coding1_Introduction)

Install virtualenv
```
Unix/macOS: python3 -m pip install --user virtualenv
Windows: py -m pip install --user virtualenv
```

Create a virtual environment
 ```
 Unix/macOS: python3 -m venv environment_name
 Windows: py -m venv environment_name
 ```
 activate the virtual environment
 ```
 Unix/macOS: source environment_name/bin/activate
 Windows: .\environment_name\Scripts\activate
 ```
 and install relevant dependencies that are listed in the requirements.txt file
 ```
 pip install -r requirements.txt
 ```
 
 NOTE: There might be some error flags related to CUDA when you start running the programming. You can ignore these. If the program starts to print out episode and total reward, everything is fine.
