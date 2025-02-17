import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import classification_report
from clustering import *
from classification import *

print("\n********CLUSTERING USING K MEANS******\n\n")
energy_clustering()

print("\n\n\nCHICAGO FOOD INSPECTIONS: CLASSIFICATION USING DECISION TREE AND LOGISTIC REGRESSION ")
classifyFoodInspectionReports()