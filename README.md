# Predicting-House-Prices-in-Philadelphia
Machine Learning Project: Predicting House Prices in Philadelphia

# Data Collection
In the initial stages of my machine learning project aimed at predicting house prices in Philadelphia, I started with the data collection and preprocessing phases. For data collection, I compiled a dataset from real estate websites, public databases, and APIs, ensuring it included a diverse range of house characteristics such as location, size, and other relevant features. This dataset comprised over 5,000 entries, which I believed would be sufficient for developing a robust model.
![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/d0047513-9d30-4da9-830d-c8b1903b8619)

# Data Preprocessing
## Before Preprocessing 
Before preprocessing, the dataset comprised a raw combination of house details from various sources. It included critical information such as location, size, age, number of bedrooms, and more, vital for predicting house prices. However, this raw form posed significant challenges: numerous missing values across different fields disrupted the continuity of data, outliers distorted the true distribution of house prices, and categorical variables were in text form, necessitating conversion to a numerical format for machine learning algorithms. The dataset's state made it unsuitable for direct application in predictive modeling, necessitating comprehensive preprocessing to address these issues.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/a1fa7d28-a9e0-43f1-808a-ef1c3b1007df)

## After Preprocessing

### Missing Value Imputation
The preprocessing began with addressing missing values in significant columns like 'central_air' and 'basements'. The 'central_air' column, which indicates the presence of central air conditioning, had missing values filled with 'N'. Similarly, the 'basements' column, which describes the type of basement a property has, saw its missing values replaced with 'None', indicating properties without basements. This approach ensures that the model accounts for properties lacking these features instead of discarding or misinterpreting their data due to missing values.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/0d615dfc-d8a8-4ace-b66f-ebdb1370b2e5)

### Feature Selection
For model training, a subset of features was selected based on their presumed relevance to the target variable, 'sale_price'. The chosen features included 'total_livable_area', the newly created 'log_market_value', and 'number_of_bathrooms'. This selection process focuses the model on key predictors, reducing complexity and potentially improving interpretability and performance.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/b075b174-ea56-4ccd-a4a0-a99533078626)

### Dataset Splitting & Dropping Unnecessary Columns
The dataset was then split into training and testing sets, with 20% of the data reserved for testing. This split is crucial for evaluating the model's performance on unseen data, ensuring that the assessments of its predictive accuracy are realistic and reliable. Also, The 'unit' column was dropped from the dataset. This step was necessary as it was filled with null values, provided little predictive value and could have introduced unnecessary complexity into the model.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/fd9deffc-37c2-407f-bd31-d7a62157d434)

### Outlier Removal
Outlier detection and removal were performed on numeric columns using the Interquartile Range (IQR) method. This method identified outliers as those values lying outside 1.5 times the IQR from the first and third quartiles. Rows containing these outliers were removed from the dataset, resulting in a cleaner dataset ('df_clean') that's less likely to be skewed by extreme values. The comparison between the original and cleaned DataFrame shapes illustrates the extent of outlier removal, highlighting a commitment to enhancing data quality.

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
Recognizing the potential impact of a property's age on its market value, I introduced another critical feature: 'house_age'. I calculated this feature by subtracting the 'year_built' of each house from the last year which was 2023, providing me with the age of the house in years. This new feature aimed to capture the depreciation or appreciation effects associated with the age of properties, under the assumption that newer homes might fetch higher prices due to less wear and tear, more modern designs, and up-to-date amenities. Conversely, older homes might have historical value or may require more maintenance, factors that could also significantly influence their selling prices.

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

### Results and Insights
The best parameters identified were:

- "n_estimators: 182", suggesting that a relatively high number of trees in the forest was beneficial for capturing the complex relationships in the data.
- "max_depth: 30", indicating an optimal depth that was neither too shallow (potentially missing out on capturing relevant patterns) nor too deep (which could lead to overfitting).
- "min_samples_split: 8", a value that strikes a balance between allowing sufficient data points in each node for meaningful splits and preventing overly granular splits that do not generalize well.
The best RMSE score achieved through this optimized configuration was 279,487.545, representing a significant improvement over the initial model's performance. This process underscored the importance of hyperparameter tuning in machine learning workflows, demonstrating how a systematic search across a defined hyperparameter space could markedly enhance model accuracy and robustness.

By integrating RandomizedSearchCV into my model development process, I not only optimized my RandomForestRegressor model's hyperparameters but also gained valuable insights into its behavior and performance characteristics. This optimization step was instrumental in fine-tuning my model to better predict house prices in Philadelphia, showcasing the practical benefits of leveraging advanced machine learning techniques in real-world applications.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/d4d792ea-1619-4d66-ae2f-7db27ff058ac)

# Construction of Test Dataset, Evaluation of the Model & Visualizations

In the final phase of my project on predicting house prices in Philadelphia, I focused on constructing a test dataset to rigorously evaluate the performance of my optimized model. This step was critical to ensure the model's predictions were not only accurate but also generalizable to new, unseen data. The test dataset comprised the most recent data on house sales, which allowed me to assess how well the model could predict prices in the current market environment.

## Model Training with Optimized Hyperparameters
After identifying the best hyperparameters through RandomizedSearchCV—182 trees, a maximum depth of 30, and a minimum sample split of 8—I trained the RandomForestRegressor model with these optimized settings. This process aimed to refine the model's ability to understand and predict the complex dynamics of the Philadelphia housing market.

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/7cfb326a-43d5-4d09-b50a-1a11c310d206)

## Test Dataset Evaluation
Upon applying the optimized model to the test dataset, I calculated two essential evaluation metrics: the Root Mean Squared Error (RMSE) and the coefficient of determination (R^2). The optimized model achieved an RMSE of 298,128.092 and an R^2 of 0.436. These metrics provided a quantitative measure of the model's predictive accuracy and the proportion of variance in house prices that the model could explain, respectively.

- The RMSE of 298,128.092, while significant, indicated that on average, the model's predictions were within this range of the actual sale prices. Considering the vast variability in house prices and the myriad of factors influencing them, this level of accuracy is a promising result for such a complex regression task.

- The R^2 value of 0.436 further underscored the model's effectiveness, demonstrating that nearly 44% of the variability in house prices was accounted for by the model. This is particularly encouraging given the challenging nature of accurately predicting real estate prices, which are influenced by both quantifiable attributes (like square footage and the number of bathrooms) and more subjective factors (such as the desirability of the location and the quality of local schools).

## Feature Importance Analysis
A critical component of my evaluation was understanding which features contributed most significantly to the model's predictions. By examining the feature importances generated by the RandomForestRegressor, I gained valuable insights into the drivers of house prices in Philadelphia. The analysis revealed that 'log_market_value' was the most influential feature, with an importance score of approximately 67%, followed by 'total_livable_area' (around 29%) and 'number_of_bathrooms' (about 4%).

![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/6fdf7c28-97d9-4042-8525-001298a781c2)


This distribution of feature importances highlights the crucial role of a property's market value and size in determining its sale price. It also points to the relevance of the number of bathrooms, albeit to a lesser extent. These findings align with intuitive expectations about the real estate market, where a property's perceived value and its size are primary determinants of its price.

# Visualization

## Scatter Plot
![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/d90eaa0f-fd63-4a8e-8f4b-0fb4f1ffe27a)

The scatter plot provided displays the relationship between the actual sale prices and the predicted sale prices of houses from the machine learning model I developed. This visualization is an effective tool for evaluating the performance of the model.

The data points are represented by blue dots, each dot signifying an individual house's actual sale price against the model's prediction for that same house. The red dashed line represents the line of perfect prediction; if a point lies on this line, it means the predicted sale price equals the actual sale price. Here are some positive observations from the plot:

1. **Cluster Around the Line of Perfect Prediction**: Many blue dots cluster near the red dashed line, especially at the lower end of the sale price range. This indicates that the model's predictions are quite accurate for a substantial number of houses, particularly those at the lower to mid-range of prices.

2. **Symmetry in Distribution**: The predictions above and below the line seem symmetrically distributed, implying that the model does not have a consistent overestimation or underestimation bias. This symmetry suggests that the model's errors are balanced, a desirable characteristic in predictive modeling.

3. **Performance Across Different Price Ranges**: While there is greater variance in predictions for higher-priced houses, this is to be expected due to the higher variability and lower frequency of such properties in the dataset. Despite this, the model still captures the trend even in the higher-end market, as seen by some of the data points aligning closely with the line of perfect prediction in this range.

4. **R^2 Value**: An R^2 of 0.436 implies that the model can explain approximately 43.6% of the variance in the sale prices, which, given the complexity of the real estate market, is a promising result. It indicates that the model has a good predictive power for a market where price determinants are multifaceted and often interrelated.

5. **RMSE**: The RMSE value provides a straightforward measure of the model's average error magnitude. Considering the scale of house prices, the RMSE indicates that the model's predictions are reasonably close to the actual values, which is commendable for a regression problem with such diverse and dynamic data.

Overall, the scatterplot supports the evaluation that the model is a useful tool for predicting house prices in Philadelphia. It seems to perform well, particularly in the mid-range of house prices, which represents a significant portion of the market. As the model continues to learn from more data, especially at the higher end of the market, its predictions may become even more accurate. These promising results offer a strong justification for the model's deployment in a real-world setting, where it could assist buyers, sellers, and realtors in making informed decisions about property values.

## Bar Chart
![image](https://github.com/Kunj-13/Predicting-House-Prices-in-Philadelphia/assets/143433713/246a29ec-2728-43b7-9d1c-19c7bb16962e)

The bar chart I created illustrates the comparison of actual versus predicted sale prices across different price ranges for houses. The bins created represent various price intervals (<$100k, $100k-$200k, $200k-$300k, $300k-$400k, >$400k), which help in analyzing the model's performance across the housing market spectrum.

My Observations from the chart include:

1. Consistency Across Price Ranges: The model's predictions are consistently close to the actual values across different price ranges, as shown by the proximity of the red 'Predicted' bars to the blue 'Actual' bars. This suggests that the model is reliable and performs uniformly across various market segments.
2. Higher Variance at Higher Prices: As the price range increases, there appears to be a larger variance in predictions, indicated by the error bars. This is expected and aligns with the real-world scenario where higher-priced houses can have a wide range of unique features that are challenging to capture in a model.
3. Strong Performance in Mid-Range Market: The model performs particularly well in the mid-range market segments ($100k-$300k), where the majority of real estate transactions typically occur. This can be particularly useful for real estate stakeholders interested in the most active segments of the market.
4. Market Insight: The alignment between actual and predicted values, especially in the most populated price bins, provides insight into the market dynamics. It suggests that the model has learned the key features that drive house prices in these segments.

# Conclusion/Reflection

As I reflect on the completion of my machine learning project to predict the selling prices of houses in Philadelphia, it is clear that the project has been both challenging and enlightening. The project allowed me to delve deep into the real-world application of predictive analytics and provided me with invaluable practical experience in handling, processing, and modeling complex datasets.

## Challenges Encountered
- Throughout the model development phase, several challenges presented themselves. The most significant was dealing with missing or incomplete data, which is a common issue in real-world datasets. Determining the most appropriate way to handle these missing values without introducing bias or losing critical information required careful consideration and experimentation.
- Another challenge was the selection and tuning of the model. The initial models provided a decent baseline for prediction, but they were far from perfect. Deciding which features to include, which model to use, and how to optimize its parameters were decisions that required a balance of theoretical knowledge and practical experimentation.

## Recommendations for future Improvement
Based on the project findings, I would implement these few strategies in the future to improve the predictive performance of the model further:

- **Incorporating More Granular Data**: Adding more detailed attributes about the properties, such as the quality of finishes, exact location data, and economic indicators, could provide the model with a richer context for making predictions.
- **Advanced Feature Engineering**: Applying more sophisticated feature engineering techniques, such as natural language processing to analyze the textual descriptions of listings, could uncover subtle factors that influence price.
- **Ensemble and Hybrid Models**: Exploring ensemble methods that combine the predictions of multiple models or using hybrid approaches that integrate machine learning with rules-based systems might yield better performance.
- **Deep Learning**: With sufficient data, deep learning models could potentially capture complex, non-linear relationships that traditional models might miss.

## Strengths and Limitations
The strength of my approach lay in its systematic methodology—from comprehensive data preprocessing to thoughtful feature engineering and rigorous model evaluation. The use of **RandomForestRegressor** and the optimization of its hyperparameters allowed me to model non-linear relationships and feature interactions effectively.

However, limitations were also evident. The model's accuracy varied significantly across different price ranges, and the R^2 value, while reasonable, indicated that there was still a substantial amount of variability left unexplained.

## Reflection and Future Research Directions
The project underscored the dynamic nature of the real estate market, where numerous variables interact in complex ways to influence house prices. For future research, I would follow these directions:

- **Time Series Analysis**: Considering the temporal dynamics of the housing market, a time series analysis could provide insights into how prices evolve and fluctuate over time.
Exploring Alternative Data Sources: Utilizing data from emerging markets, such as short-term rental prices or real estate investment trends, could offer additional predictive power.
- **Geospatial Analysis**: Integrating geospatial information systems (GIS) data could allow the model to account for location-based factors more accurately.
In conclusion, while challenges were present, the project has laid a solid foundation for predicting house prices and highlighted several pathways for future enhancements. The experience has not only bolstered my data science skills but also reinforced the value of persistence and creativity in tackling complex, real-world problems.

