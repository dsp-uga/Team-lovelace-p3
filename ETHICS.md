


# Data Science Ethics Checklist

## A. Data Collection

 - [] **A.1 Informed consent**: If there are human subjects, have they given informed consent, where subjects affirmatively opt-in and have a clear understanding of the data uses to which they consent?

 
 - [x] **A.2 Collection bias**: Have we considered sources of bias that could be introduced during data collection and survey design and taken steps to mitigate those? 
 Yes, we considered racial and gender bias. And we collected data from diverge backgrounds. 
 
 - [x] **A.3 Limit PII exposure**: Have we considered ways to minimize exposure of personally identifiable information (PII) for example through anonymization or not collecting information that isn't relevant for analysis?
 This dataset does not use PII information. We remove subject names and their identity in the dataset collection, just maintaining one or two personal traits which can never be used to identity an individual solely such as ethinicity and gender. 

## B. Data Storage
 - [x] **B.1 Data security**: Do we have a plan to protect and secure data (e.g., encryption at rest and in transit, access controls on internal users and third parties, access logs, and up-to-date software)?
 
 We will protect this data against any misuse by giving access to only machine learning researcher of this field. 
 
 - [x] **B.2 Right to be forgotten**: Do we have a mechanism through which an individual can request their personal information be removed?
 The data does not contain PII exposure. However, we reserve the right for every individual. They can share their index number with us and we can remove their data from our dataset. 
 - [] **B.3 Data retention plan**: Is there a schedule or plan to delete the data after it is no longer needed?
    
     
## C. Analysis
 - [] **C.1 Missing perspectives**: Have we sought to address blindspots in the analysis through engagement with relevant stakeholders (e.g., checking assumptions and discussing implications with affected communities and subject matter experts)?
 - [] **C.2 Dataset bias**: Have we examined the data for possible sources of bias and taken steps to mitigate or address these biases (e.g., stereotype perpetuation, confirmation bias, imbalanced classes, or omitted confounding variables)?
 - [] **C.3 Honest representation**: Are our visualizations, summary statistics, and reports designed to honestly represent the underlying data?
 - [x] **C.4 Privacy in analysis**: Have we ensured that data with PII are not used or displayed unless necessary for the analysis?
 Yes, we have removed all PII. The only information that remains is images of neurons firing across a time scale. 
 
 - [x] **C.5 Auditability**: Is the process of generating the analysis well documented and reproducible if we discover issues in the future?
 Yes, we have created a thorough documentation to reproduce our results. 

## D. Modeling
 - [x] **D.1 Proxy discrimination**: Have we ensured that the model does not rely on variables or proxies for variables that are unfairly discriminatory?
     No variables are discriminatory. The firing of neurons in brain is a general phenomenon among humans.  
 - [] **D.2 Fairness across groups**: Have we tested model results for fairness with respect to different affected groups (e.g., tested for disparate error rates)?
 - [] **D.3 Metric selection**: Have we considered the effects of optimizing for our defined metrics and considered additional metrics?
 - [] **D.4 Explainability**: Can we explain in understandable terms a decision the model made in cases where a justification is needed?
 - [] **D.5 Communicate bias**: Have we communicated the shortcomings, limitations, and biases of the model to relevant stakeholders in ways that can be generally understood?

## E. Deployment
 - [x] **E.1 Redress**: Have we discussed with our organization a plan for response if users are harmed by the results (e.g., how does the data science team evaluate these cases and update analysis and models to prevent future harm)?
 The only way users can be harmed by this result is if they have a medical condition that we found and is leaked by us. But since we do not hold data with personal information. Users can not be harmed by results. Moreover, this study only calculate calcium ions based firing of neurons. It can not detect any diseases or ailments with certainity according to current medical knowledge. 
 - [] **E.2 Roll back**: Is there a way to turn off or roll back the model in production if necessary?
 - [] **E.3 Concept drift**: Do we test and monitor for concept drift to ensure the model remains fair over time?
 - [] **E.4 Unintended use**: Have we taken steps to identify and prevent unintended uses and abuse of the model and do we have a plan to monitor these once the model is deployed?

*Data Science Ethics Checklist generated with [deon](http://deon.drivendata.org).*
