# from ..utils.utilities import Classifier
# from ..utils import utilities
from src.utils.utilities import Classifier


class TestClassifier(Classifier):
# class TestClassifier(utilities.Classifier):
  def __init__(self):
    pass

  def predict(self, x):
    pass

  def train(self, samples, labels):
    pass


if __name__ == "__main__":
  clf = TestClassifier()
  print "-----"
  print type(clf)