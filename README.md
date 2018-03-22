# ml-gui
This is a little toy program that lets you place dots of the color red, green and blue on the canvas and allows you to visualize the predicted regions of your own discriminant classifiers algorithms. Look at the images at the bottom to see how it looks like or even better: Clone the repository, fire up the program and play with it. It is a toy program after all.

Requirements: Python 2.7, numpy

`cd` to the directory where you want to clone the repository into. Clone with

`git clone https://github.com/munluk/ml-gui.git`

To startup the application run
```
cd ml-gui
python DiscriminantClassifierGUI.py
```

In order to use your own classifier you have to create a new .py file in the `src/classifiers/`. Create a class the inerherits the class `classifier` from `src/utils/utilities`. ***In order to work with the Gui, the class name must match the file name of your classifier script.***

The file `src/resources/ClassifierTemplate.py` contains a template for a classifier which you can use.

# Preview
<img src="https://github.com/munluk/ml-gui/blob/master/images/k-discriminant-classification.png" width="90%"></img>
<img src="https://github.com/munluk/ml-gui/blob/master/images/mlp_classification.png" width="90%"></img>
<img src="https://github.com/munluk/ml-gui/blob/master/images/soft-zero-one_classification.png" width="90%"></img>

