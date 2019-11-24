# Sugr Rush
Just as diabetes can affect your quality of life, your quality of life can affect your diabetes.

# Inspiration
Each of our group members have been affected by Type 2 Diabetes as it runs in our families. 
From personal experience, we understand that diabetes can be a demanding condition to manage and affect our lives in many ways. 
Everyone wants to have the best possible quality of life. It just feels good to be satisfied and happy. 
However, there is another reason, as well. Just as diabetes can affect your quality of life, your quality of life can affect your diabetes. 
This notion inspired our application: Sugr Rush, a mobile application for identification of Type 2 diabetes and voice-enabled workout routines to prevent further severity of the condition.

# What it does
Sugr Rush is a mobile application which identifies whether a female individual has Type 2 diabetes or not. 
It suggests workout routines with voice-enabled dialogue to guide patients through specified level workouts as per their health conditions related to diabetes. 
The application has a physician/doctor login and patient login.

1. On the patient side, the individual can request a diagnosis from their doctor at their clinic on the interface.

2. On the doctor side, the physician can login to his/her database and select new diagnoses to complete as per health care records. 
a) The doctor fills out patient information on the input parameters pertaining to Type 2 diabetes. 
b) The interface will output whether the individual is at low risk or high risk for Type 2 Diabetes. 
c) Information gets relayed to patient side

3. On the patient side, the individual views diagnosis and obtains workout routines tailored to their specific level of severity for diabetes. 
a) Voice-enabled workouts help the individual complete routines based on their frequency and time preferences 
b) Individuals may keep track of their workouts on interface and request updated diagnosis from doctor after completion of workouts in a set duration of time.

# How we built it
The model is built using python’s machine learning libraries (tensorflow, Keras) by incorporating deep neural networks to predict whether a patient has type 2 diabetes or not. 
The model uses the K-fold Cross Validation(CV) technique by dividing the data into folds and ensuring that each fold is used as a testing set at some point. 
The data set consists of some medical distinct variables, such as pregnancy record, BMI, insulin level, age, glucose concentration, diastolic blood pressure, triceps skin fold thickness, diabetes pedigree function etc. 
The data set has 768 patient’s data where all the patients are female and at least 21 years old.

The front-end application interface was built on Sketch and Android Studio with the server hosted on the Internet. In addition, the voice-enabled, NLP technology was integrated with Dialogflow, an API on Google Cloud.

# Challenges we ran into
1. Coming up with a solid idea that we are all passionate about
2. Connectivity issues related to integrating functionality of the back-end, front-end and speech into one application
3. Dialogflow challenges with voice integration
4. Learning curve in understanding new languages from scratch (JavaScript)

# Accomplishments that we're proud of and what we learned
1. Learning more about the implications of Type 2 Diabetes
2. Working with the Dialogflow API for the first time
3. Communication between an application and a server hosted on the internet
4. Leveraging our skill-sets while learning new skills (whether it's a new coding language or constructing an interface)

# What's next for Sugr Rush
1. Applying the solution across different demographics (males, children, individuals across different populations around the world)
2. Learning to diagnosis more specific problems related to diabetes (identification of Type 1 vs. Type 2, early detection, and hearing loss prevention from diabetes)
3. Integrating a communication network between doctor-side and patient-side for other conditions outside of diabetes on the application itself

# Built With
android-studio /
dialogflow /
flask /
google-web-speech-api /
gridsearchcv /
java /
javascript /
json /
keras /
kfoldcv /
ngrok /
numpy /
okhttp3 /
pandas /
python /
sketch / 
sklearn /
tensorflow 
