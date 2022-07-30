# Neural_Network_Charity_Analysis

## Overview of the analysis: 
The purpose of this analysis was to help Alphabet Soup - a charitable foundation predict where to make investments. I leveraged my knowledge of machine learning and neural networks, to analyze a CSV containing more than 34,000 organizations that have received funding from the foundation over the years and created a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

## Results: 

### Data Preprocessing
1. What variable(s) are considered the target(s) for your model?
- The target for the model was the 'IS_SUCCESSFUL' variable i.e. was the money used effectively

2. What variable(s) are considered to be the features for your model?
- For the first model I considered the following variables as the features:
    - APPLICATION_TYPE—Alphabet Soup application type
    -  AFFILIATION—Affiliated sector of industry
    - CLASSIFICATION—Government organization classification
    - USE_CASE—Use case for funding
    - ORGANIZATION—Organization type
    - STATUS—Active status
    - INCOME_AMT—Income classification
    - SPECIAL_CONSIDERATIONS—Special consideration for application
    - ASK_AMT—Funding amount requested
- For the second model, which performed better than the first one, I also used NAME variable as a feature for my model

3. What variable(s) are neither targets nor features, and should be removed from the input data?
- For the first model I dropped the EIN and NAME columns from the dataframe. However, in the second model, I kept the NAME column.

4. How did you preprocess the dataset to compile, train, and evaluate the neural network model?
 - Dropped the EIN and NAME columns for the first model. But only dropped the EIN column for the second model.
 - Determined the number of unique values for each column.
 - For those columns that have more than 10 unique values, determined the number of data points for each unique value.
 - Created a density plot to determine the distribution of the column values.
    
    Density Plot name_count (Model 2 only)
    <img src="/Resources/name_count_density_plot.png" >

    Density Plot application_type_count
    <img src="/Resources/application_count_density_plot.png" >

    Density Plot classification_count
    <img src="/Resources/classification_density_plot.png" >
    
 - Used the density plot to create a cutoff point to bin "rare" categorical variables together in a new column, Other, and then checked if the binning was successful.

    Binned name_count (Model 2 only)
    <img src="/Resources/name_count_binned.png" >

    Binned application_type_count
    <img src="/Resources/application_count_binned.png" >

    Binned classification_count
    <img src="/Resources/classification_binned.png" >

 - Generated a list of categorical variables: 
 
    Model 1
    <img src="/Resources/categorical_variables1.png" >

    Model 2
    <img src="/Resources/categorical_variables2.png" >

 - Encoded categorical variables using one-hot encoding, and placed the variables in a new DataFrame.
 - Merged the one-hot encoding DataFrame with the original DataFrame, and dropped the originals.

     Updated df Model 1
    <img src="/Resources/final_df1.png" >

    Updated df Model 2
    <img src="/Resources/final_df2.png" >

 - Split the preprocessed data into features and target arrays.
 - Split the preprocessed data into training and testing datasets.
 - Standardized numerical variables using Scikit-Learn’s StandardScaler class, then scaled the data.

### Compiling, Training, and Evaluating the Model
1. How many neurons, layers, and activation functions did you select for your neural network model, and why?
    Since the data size was not too big or complex, I had two hidden layers in the first model. The first hidden layer had 9 nodes and the second had 5. I used the relu activation function for both the hidden layers and used sigmoid activation function for the output layer to finally classify the data.

    Structure Model 1
    <img src="/Resources/model_structure1.png" >

    Since adding the NAME column as the feature increased the data size, I added one more hidden layer to the second model. I also increased the nodes for each layer to account for the data complexity. The first hidden layer had 15 nodes, the second had 11, and the third had 5. I used the relu activation function for all the hidden layers and used sigmoid activation function for the output layer to finally classify the data.

    Structure Model 2
    <img src="/Resources/model_structure2.png" >

2. Were you able to achieve the target model performance?

    The first model was 73.59% accurate in training but only 72.36% accurate in testing. It did not meet my target requirement of at least 75% accuracy.

    Performance Model 1
    <img src="/Resources/model_accuracy1.png" >

    The second model was 79.73% accurate in training and 77.93% accurate in testing. It met my target requirement of at least 75% accuracy.

    Performance Model 2
    <img src="/Resources/model_accuracy2.png" >

3. What steps did you take to try and increase model performance?
    To improve the performance of the model I tried a few things:
    1. I added a third hidden layer with 3 nodes and did not change anything else. This change did not achieve the target model performance.
    2. I added the third hidden layer as described above and changed the activation function for the output layer to tanh. These changes still did not achieve the target model performance.
    3. I did not drop the NAME column and used it as a feature. Also, added a third hidden layer, and increased the number of nodes for each layer. Left the activation function for the output layer as sigmoid. These changes did meet the target model performance.

## Summary: 
Overall, the deep learning models were fairly accurately able to predict whether applicants will be successful if funded by Alphabet Soup. However, I question the use of deep learning models over other linear regression models in this case due to following reasons:
 - the effort required to clean up the data
 - computation resources required to run the model
 - inability to explain the model logic

 So my recommendation is to try other linear regression models to analyze the data and see if similar or better model performance can be achieved.