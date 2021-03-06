# Neural-Network-for-Candidate-Selection
pybrain implementation ANN(Artificial Neural Network)

latest accuracy:** 0.61350575(ANN_new_new.py)

### Selected Features
0. id
1. status
2. age
3. activities num
4. language num
5. scholarship
6. secondary_education num
7. teriary_education num
8. honour
9. major_of_tertiary_education
10. qualification_of_tertiary_education
11. study_mode_of_tertiary_education
12. university_of_tertiary_education;
13. study time;
14. working_exp num;
15. working_exp total duration_months;

### Major steps
1. delete other status to consider it as an binary classification problem
2. transfer text values to numbers in some features(feature engineering)
3. run pybrain models

### Dependencies
* numpy
* scipy
* pybrain
* sklearn(for preprocessing)

### Environment
python 2.7.12 on Ubuntu 16.04

*official docs of pybrain* http://www.pybrain.org/docs/

### Tasklist
- [x] Information extracting
- [x] Raw SVM model
- [x] Raw Neural Network model
- [x] Cross Validation (Neural Network is chosen)
- [x] Feature Engineering 
    - [x] Mutual Information
    - [x] penalty function
    - [x] Manual Data Grouping 
- [x] parameter tuning  
- [ ] enhanced model
    - [ ] LSTM
    - [ ] RNN (Recurrent Neural Networks)
    
- [ ] Evaluate the forcasting result
if the result is usable:
- [ ] Embed the learning system to the web application
else:
- [ ] Summary on the failure
