# 1. Introduction to Machine Learning

### 1.1 Introduction to Machine Learning

Machine Learning is a process of extracting patterns from data, which is of two types:

- features (information about the object)
- target (property to predict for unseen objects)

Therefore, new feature values are presented to the model, and it makes predictions from the learned patterns.

### 1.2 Machine Learning vs Rule-Based Systems

The difference between ML and Rule-Based systems is explained with the example of a spam filter.

Traditional Rule-Based systems are based on a set of characteristics (keywords, email length, etc.) that identify an email as spam or not. As spam emails keep changing over time the system needs to be upgraded making the process intractable due to the complexity of code maintenance as the system grows.

ML can be used to solve this problem with the following steps:

- **Get data:** Emails from the user's spam folder and inbox give examples of spam and non-spam.
- **Define and calculate features:** Rules/characteristics from rule-based systems can be used as a starting point to define features for the ML model. The value of the target variable for each email can be defined based on where the email was obtained from (spam folder or inbox). Each email can be encoded (converted) to the values of its features and target.
- **Train and use the model:** A machine learning algorithm can then be applied to the encoded emails to build a model that can predict whether a new email is spam or not spam. The predictions are probabilities, and to make a decision it is necessary to define a threshold to classify emails as spam or not spam.

### 1.3 Supervised Machine Learning

In Supervised Machine Learning (SML) there are always labels associated with certain features. The model is trained, and then it can make predictions on new features. In this way, the model is taught by certain features and targets.

- **Feature matrix ($X$):** Made of observations or objects (rows) and features (columns).
- **Target variable ($y$):** A vector with the target information we want to predict. For each row of $X$ there's a value in $y$.

The model can be represented as a function, $g$, that takes the feature matrix, $X$, as input and tries to predict values as close as possible to the targets, $y$. The process of finding this function $g$ is called training.

Types of SML problems

- **Regression:** The output is a number (car's price).
- **Classification:** The output is a category (spam example).
  - *Binary:* There are two categories.
  - *Multiclass problems:* There are more than two categories.
- **Ranking:** The output is the top scores associated with corresponding items. It is applied in recommender systems.

In summary, SML is about teaching the model by showing it different examples, and the goal is to come up with a function, that takes the feature matrix as input, and makes predictions of values as close as possible to the $y$ targets.

### 1.4 CRISP-DM

CRISP-DM, which stands for Cross-Industry Standard Process for Data Mining, is an open standard process model that describes common approaches used by data mining experts. It is the most widely-used analytics model.

1. **Business understanding:** An important question is do we need ML for the project. The goal of the project has to be measurable.
2. **Data understanding:** Analyze available data sources, and decide if more data is required.
3. **Data preparation:** Clean data, remove noise applying pipelines, and convert the data to a tabular format, so we can put it into ML.
4. **Modeling:** Train different models and choose the best one. Considering the results of this step, it is proper to decide if it is required to add new features or fix data issues.
5. **Evaluation:** Measure how well the model is performing and if it solves the business problem.
6. **Deployment:** Roll out to production to all the users. The evaluation and deployment often happen together - online evaluation.

### 1.5 Model Selection Process

Which model to choose?

- Logistic regression
- Decision tree
- Neural Network
- Or many others

The validation dataset is not used in training. There are feature matrices and y vectors for both training and validation datasets. The model is fitted with training data, and it is used to predict the y values of the validation feature matrix. Then, the predicted y values (probabilities) are compared with the actual y values.

**Multiple comparisons problem (MCP):** Just by chance one model can be lucky and obtain good predictions because all of them are probabilistic. The test set can help to avoid the MCP. Obtaining the best model is done with the training and validation datasets, while the test dataset is used for assuring that the proposed best model is the best.

1. Split datasets in training, validation, and test. E.g. 60%, 20% and 20% respectively
2. Train the models
3. Evaluate the models
4. Select the best model
5. Apply the best model to the test dataset
6. Compare the performance metrics of validation and test
   
NB: Note that it is possible to reuse the validation data. After selecting the best model (step 4), the validation and training datasets can be combined to form a single training dataset for the chosen model before testing it on the test set.

### 1.6 Setting up the Environment

pass

### 1.7 Introduction to NumPy

pass

### 1.8 Linear Algebra Refresher

pass

### 1.9 Introduction to Pandas

pass
