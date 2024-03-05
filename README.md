# Predicting-House-Prices-in-Philadelphia
Machine Learning Project: Predicting House Prices in Philadelphia

# Data Collection
In the initial stages of my machine learning project aimed at predicting house prices in Philadelphia, I started with the data collection and preprocessing phases. For data collection, I compiled a dataset from real estate websites, public databases, and APIs, ensuring it included a diverse range of house characteristics such as location, size, and other relevant features. This dataset comprised over 5,000 entries, which I believed would be sufficient for developing a robust model.
![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/d0047513-9d30-4da9-830d-c8b1903b8619)

# Data Preprocessing
## Before Preprocessing 
Before preprocessing, the dataset comprised a raw amalgam of house details from various sources. It included critical information such as location, size, age, number of bedrooms, and more, vital for predicting house prices. However, this raw form posed significant challenges: numerous missing values across different fields disrupted the continuity of data, outliers distorted the true distribution of house prices, and categorical variables were in text form, necessitating conversion to a numerical format for machine learning algorithms. The dataset's state made it unsuitable for direct application in predictive modeling, necessitating comprehensive preprocessing to address these issues.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/a1fa7d28-a9e0-43f1-808a-ef1c3b1007df)

## After Preprocessing

### Missing Value Imputation
The preprocessing began with addressing missing values in significant columns like 'central_air' and 'basements'. The 'central_air' column, which indicates the presence of central air conditioning, had missing values filled with 'N', assuming the absence of this feature where not explicitly mentioned. Similarly, the 'basements' column, which describes the type of basement a property has, saw its missing values replaced with 'None', indicating properties without basements. This approach ensures that the model accounts for properties lacking these features instead of discarding or misinterpreting their data due to missing values.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/0d615dfc-d8a8-4ace-b66f-ebdb1370b2e5)

### Feature Selection
For model training, a subset of features was selected based on their presumed relevance to the target variable, 'sale_price'. The chosen features included 'total_livable_area', the newly created 'log_market_value', and 'number_of_bathrooms'. This selection process focuses the model on key predictors, reducing complexity and potentially improving interpretability and performance.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/b075b174-ea56-4ccd-a4a0-a99533078626)

### Dataset Splitting & Dropping Unnecessary Columns
The dataset was then split into training and testing sets, with 20% of the data reserved for testing. This split is crucial for evaluating the model's performance on unseen data, ensuring that the assessments of its predictive accuracy are realistic and reliable. Also, The 'unit' column was dropped from the dataset. This step likely reflects a decision to exclude features that either provide little predictive value or could introduce unnecessary complexity into the model.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/fd9deffc-37c2-407f-bd31-d7a62157d434)

### Outlier Removal
Outlier detection and removal were performed on numeric columns using the Interquartile Range (IQR) method. This method identifies outliers as those values lying outside 1.5 times the IQR from the first and third quartiles. Rows containing these outliers were removed from the dataset, resulting in a cleaner dataset ('df_clean') that's less likely to be skewed by extreme values. The comparison between the original and cleaned DataFrame shapes illustrates the extent of outlier removal, highlighting a commitment to enhancing data quality.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/d42a5817-be76-4ea8-8c94-9529483fb22a) 

## Final Results of Preprocessing
These preprocessing steps helped refine the dataset, making it more amenable to effective model training. By imputing missing values, selecting relevant predictors, splitting the data appropriately, and removing outliers, the preprocessing phase helped lay a solid foundation for building a predictive model.  

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/785acdf9-5266-48c7-8028-21f9a1869c8b)

# Feature Engineering

This feature engineering phase was meticulously designed to enhance my dataset's complexity and depth, enabling my models to capture a more nuanced understanding of what drives house prices in Philadelphia. By incorporating polynomial features, I was able to model the relationship between house size and area in a more flexible and informative manner, potentially capturing synergistic effects that linear terms alone could miss. The addition of 'house_age' introduced a direct measure of a property's age, a factor known to influence buyer preferences and, consequently, house prices.

## Polynomial Features
I initiated the feature engineering process by focusing on two key numerical variables: 'total_livable_area' and 'total_area'. My hypothesis was that the interactions between these features, as well as their squared terms, could provide deeper insights into their combined effect on house prices. To test this hypothesis, I utilized the PolynomialFeatures class from scikit-learn, setting the degree to 2 to include both the interaction terms and the squared terms of the original features. This approach allowed me to model nonlinear relationships without assuming a specific form for the underlying relationship between the features and the target variable.

After generating these polynomial features, I created a new DataFrame, poly_df, to hold the generated features, which included the original features, their squared terms, and the interaction term between them. I used the get_feature_names_out method to maintain readability and traceability of these new variables, ensuring they were clearly labeled according to the original features they were derived from. Subsequently, I concatenated poly_df with my original DataFrame, significantly enriching my dataset with these newly engineered features.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/eef9235f-b98f-4221-a7b0-1d53d26c56de)

## Addition of House Age Feature
Recognizing the potential impact of a property's age on its market value, I introduced another critical feature: 'house_age'. I calculated this feature by subtracting the 'year_built' of each house from the current year (2023), providing me with the age of the house in years. This new feature aimed to capture the depreciation or appreciation effects associated with the age of properties, under the assumption that newer homes might fetch higher prices due to less wear and tear, more modern designs, and up-to-date amenities. Conversely, older homes might have historical value or may require more maintenance, factors that could also significantly influence their selling prices.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/97773249-11e6-4f04-9bde-32b496bd552d)

# Model Development
## Initial Model Training
I chose to use the "RandomForestRegressor" from scikit-learn as my initial model, considering its robustness and ability to handle complex interactions between features without the need for extensive hyperparameter tuning. The RandomForestRegressor is an ensemble method that operates by constructing a multitude of decision trees at training time and outputting the average prediction of the individual trees. This approach helps in reducing overfitting and is effective for regression tasks.

I initialized the RandomForestRegressor with 100 trees (n_estimators=100) and a random state of 42 to ensure reproducibility of the results. After training the model on my training dataset (X_train, y_train), I proceeded to evaluate its performance on the test set (X_test, y_test).

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/8f13a5ec-f282-42ab-bf48-673f4765dd86)

## Model Evaluation
### RandomizedSearchCV
My approach involved defining a parameter distribution to guide the randomized search. The parameters I chose to optimize were:

- 'n_estimators': The number of trees in the forest. I considered a range from 100 to 200 trees, providing a broad spectrum to identify the optimal number of trees that balances model complexity and computational efficiency.
- 'max_depth': The maximum depth of the trees. This parameter was varied from 10 to 50, allowing the model to explore a range of tree depths to prevent overfitting (with too high a depth) or underfitting (with too low a depth).
- 'min_samples_split': The minimum number of samples required to split an internal node. By sampling values from 2 to 10, I aimed to find a sweet spot that ensures each node is split in a manner that contributes meaningfully to the model's predictive capability without causing excessive tree complexity.
I initialized the RandomizedSearchCV object with my RandomForestRegressor model, setting "n_iter=10" to limit the search to 10 iterations, each with a randomly selected combination of hyperparameters from the specified distributions. I chose a 5-fold cross-validation strategy to ensure each configuration was thoroughly evaluated against different subsets of the data, minimizing the risk of overfitting. The scoring was set to 'neg_root_mean_squared_error' to directly optimize for lower RMSE values, aligning with my project's objective of minimizing prediction errors.

Results and Insights
The best parameters identified were:

- "n_estimators: 182", suggesting that a relatively high number of trees in the forest was beneficial for capturing the complex relationships in the data.
- "max_depth: 30", indicating an optimal depth that was neither too shallow (potentially missing out on capturing relevant patterns) nor too deep (which could lead to overfitting).
- "min_samples_split: 8", a value that strikes a balance between allowing sufficient data points in each node for meaningful splits and preventing overly granular splits that do not generalize well.
The best RMSE score achieved through this optimized configuration was 279,487.545, representing a significant improvement over the initial model's performance. This process underscored the importance of hyperparameter tuning in machine learning workflows, demonstrating how a systematic search across a defined hyperparameter space could markedly enhance model accuracy and robustness.

By integrating RandomizedSearchCV into my model development process, I not only optimized my RandomForestRegressor model's hyperparameters but also gained valuable insights into its behavior and performance characteristics. This optimization step was instrumental in fine-tuning my model to better predict house prices in Philadelphia, showcasing the practical benefits of leveraging advanced machine learning techniques in real-world applications.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/d4d792ea-1619-4d66-ae2f-7db27ff058ac)

