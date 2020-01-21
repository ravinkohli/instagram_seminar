# instagram_seminar

## Requirements
1. keras
2. tensorflow
3. opencv
4. matplotlib
5. graphviz

## How to use
In the central directory, 

**Training**

```python TaskA/Train.py``` 

this will train the model and save it in the folder ./models with the timestamp appended to "model" for the name. It will also create a 
plot of the training and validation curves in ./plots. Also, it will also save the graph of netwok architecture in the same folder.

**Evaluation**

```python TaskA/Evaluate.py```

By default it will evaluate the latest model with the validation data and report the MAE and MSE.
