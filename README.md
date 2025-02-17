# Classification of Food Inspections and Prediction of Energy Benchmarking Data in Chicago

Riya N. Bhutada
Northeastern Illinois University, rnbhutada@neiu.edu
December 2023

### 🎯 Goal
- I set out to examine the food inspection reports in Chicago based on Facility Types, Risk of facility that depends on how frequent the inspections should be carried, number of violations the report specifies, and other violations related binary attributes. I did this using <b>decision trees and logistic regression</b> to classify whether the inspection report passed or failed. I found that using decision trees, the accuracy score was 0.936 and 0.952 for logistic regression. However, the train score for decision tree came out to be 0.969, which was more than test score. 
<br/><br/>
- I also set out to examine the Energy Benchmarking 2021 reported in 2022 data of Chicago based mainly on the Type of Property, Source EUI(Energy Use Intensity) used, Site EUI(Energy Use Intensity), Weather Source EUI, Weather Site EUI, ENERGY STAR score that assesses a property’s overall energy performance and Energy Rating(the-zero-to-four-star-rating). I did this using <b>K-Means clustering algorithm</b> to see if there were patterns in Energy Rating. When plotted Source EUI against Site EUI, I found clusters of data points that closely represented the Rating values from 0 to 4.

### 📝 Dataset
- #### Chicago Food Inspections Dataset <br/>
  This information is derived from inspections of restaurants and other food establishments in Chicago from January 1, 2010 to the present. Inspections are performed by the Chicago Department of Public Health’s Food Protection Program.
- #### Chicago Energy Benchmarking 2021, Data Reported in 2022 Dataset <br/>
  The dataset contains demographic information, Year, ID, Property Name, Reporting Status, Community Area, Exempt from Energy Rating, Year Built, All other fuel Use (annual amount of fuel use other than electricity), Total GHG Emissions(Total Green House Gas Emissions), GHG Intensity(Emissions/Gross Floor Area), District Chilled Water Use, Natural Gas Use, Electricity Use.

### ⚙️ How did I achieve this? 
- I examined the Food Inspection reports dataset based on Facility Types, Risk of facility (that depends on how frequent the inspections should be carried), number of violations the report specifies, and other violations related binary attributes (calculated from Violations column) as mentioned in Dataset table.
- Used Scikit library’s DecisionTreeClassifier Class and LogisticRegression class to classify the Results target attribute into Pass (i.e. 1) or Fail (i.e. 0) category.
- Examined the Energy Benchmarking 2021 reported in 2022 data of Chicago based on the Type of Property, Source EUI(Energy Use Intensity) used, Site EUI(Energy Use Intensity), Weather Source EUI, Weather Site EUI, ENERGY STAR score and Energy Rating.
- Used Scikit library’s KMeans clustering algorithm to find patterns on how the data distributes across the energy parameters. For preparing and handling the data, I used Pandas and Numpy libraries.
- Since DecisionTreeClassifier and LogisticRegression learners work with binary or numerical data, I chose ordinal encoding to handle all categorical data. The models predicted better with ordinal encoding over one-hot encoding. For data visualization, I used plotly, matplotlib and seaborn.

### 📊 Results
From below tables, Linear Regression shows a fair accuracy score and CLF score on both train and test data. For out of total 1349 test records, 1130 are positives. Recall score for both classifiers is 0.98 and 1.0, which means they predicted 97% of positive records correctly with a precision of 0.95 for positive class.

<table align="center">
    <tr>
        <th>Model</th>
        <th>Mean Absolute Error</th>
        <th>Accuracy Score</th>
        <th>CLF Score</th>
    </tr>
    <tr>
        <td>Decision Tree</td>
        <td>0.063</td>
        <td>0.936</td>
        <td>Train 0.969, Test 0.936</td>
    </tr>
    <tr>
        <td>Linear Regression</td>
        <td>0.047</td>
        <td>0.952</td>
        <td>Train 0.956, Test 0.952</td>
    </tr>
</table>
<br/>
<table align="center">
    <tr>
        <th>Output Label</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1 Score</th>
        <th>Support</th>
    </tr>
    <tr>
        <td>Fail (0)</td>
        <td>0.86</td>
        <td>0.73</td>
        <td>0.79</td>
        <td>219</td>
    </tr>
    <tr>
        <td>Pass (1)</td>
        <td>0.95</td>
        <td>0.98</td>
        <td>0.96</td>
        <td>1130</td>
    </tr>
    <tr>
        <td>Fail (0)</td>
        <td>0.98</td>
        <td>0.72</td>
        <td>0.83</td>
        <td>219</td>
    </tr>
    <tr>
        <td>Pass (1)</td>
        <td>0.95</td>
        <td>1.00</td>
        <td>0.97</td>
        <td>1130</td>
    </tr>
</table>

Overall, LinearRegression model predicted values slightly better than DecisionTree.<br/>  
The clusters predicted by K-means model show similarity with the Energy Score distribution with respect to the Energy Use Intensity features. From the right figure, for a Site and source EUI of 50kBtu, the bottom most cluster represents an Energy Score of close to 100. We can conclude, lower the Energy Use, higher is the Energy Score and better is the performance.     

<div align="center">
  <h6 align="center">Figure 1: Site EUI VS Source EUI, Energy Score on left plot VS Clusters on right plot.</h6>
  <img width="30%" src="https://github.com/user-attachments/assets/ddf2f7db-104a-4121-85e2-c8be492b3202" />
  <img width="30%" src="https://github.com/user-attachments/assets/cce92b23-ffe8-4d03-aba3-1870d5a018c2" />
</div>

<div align="center">
  <h6 align="center">Figure 2: Floor Area VS Weather Normalized Source EUI, Energy Score on left plot VS Clusters on right plot.</h6>
  <img width="30%" src="https://github.com/user-attachments/assets/91c5413b-3eb5-4418-bb80-a93daad5a7b0" />
  <img width="30%" src="https://github.com/user-attachments/assets/54c8b191-8b50-4241-be59-d4849885187a" />
</div>
<br/>

Figure 2: Floor Area VS Weather Normalized Source EUI, Energy Score on left plot VS Clusters on right plot
Fig 2 also shows clusters formed could predict Energy Rating. And that clusters with low Gross Floor Area and low Weather Normalized EUI tend to have high Energy Rating.
<br/>

### ✅ Conclusion
The lowest cluster from Fig 1 seems to contain all the buildings with low energy use, high Energy Score and therefore highest Energy Rating of 4.0. The uppermost cluster seems to be formed of high Site EUI, high Source EUI and lowest Energy Score. But, the left diagram shows some points with high energy use and close to high Energy Score. These could be outliers with other energy data affecting its Score. Mostly, all clusters seem to fit the Energy Score ranges.<br/>
<br/>Both classifiers learned and predicted the positive records pretty well. This could also be because of the total positive records present in the data set being much more than the negative class samples. Decision Tree classifier seemed to overfit the data a little since the test score was less than training score. Logistic Regression performed better with overall good metrics. To improvise on the metrics, more data could be collected, so that the model learns well and increases Precision and Recall Score.
