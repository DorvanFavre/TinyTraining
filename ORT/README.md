# Onnx Runtime Training (ORT)

In order to run the ORT pipeline, use a python virtual environement (venv) with the ORT dependecies installed.
Then, choose a task among the available ones it the folder.

## Create the environement
Create the python environement and isntall the requirements (requirements.txt)
```python
python -m venv venv 
source venv/bin/activate 
pip install -update pip
pip install -r requirements.txt
```

If the Onnxruntime isn't available for your device, you may need to build the wheel.
https://onnxruntime.ai/docs/build/training.html

## Activate the environement
```python
source venv/bin/activate 
```

If using VSCode, make it available as kernel
ctrl + shit + p
Python: select interpreter
Enter interpreter path
venv/bin/python
now you can select the intepreter
