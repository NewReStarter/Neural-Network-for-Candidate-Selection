#Training set accuracy: [0.39777219]
#Testing set accuracy: [0.37931034]
import copy
import numpy as np
# pybrain: https://github.com/pybrain/pybrain 
# python setup.py install to install pybrain
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure import TanhLayer
from pybrain.structure import MDLSTMLayer
# pybrain official site: http://www.pybrain.org/docs/
# pybrain ref I use: http://blog.csdn.net/qtlyx/article/details/51615104

dataname = r'data.csv'
data = [line.strip().split(',') for line in open(dataname, 'rb')]
del data[0]
afterjson = []
for i in range(len(data)):
    afterjson.append([])
    temp = ''
    for j in range(3, len(data[i]) - 1):
        temp = temp + ',' + data[i][j]
    temp = temp[2 : -1].replace('""', '"')

    afterjson[i].extend([data[i][2], data[i][-1], data[i][1], json.loads(temp)])
selectdata = []
for big in afterjson:
    item = big[3]
    selectdata.append([])
    selectdata[-1].append(int(big[0]))
    selectdata[-1].append(big[1])
    selectdata[-1].append(int(big[2]))
    selectdata[-1].append(len(item['activities']))
    selectdata[-1].append(len(item['language']))
    selectdata[-1].append(len(item['scholarship']))
    selectdata[-1].append(len(item['secondary_education']))
    selectdata[-1].append(len(item['tertiary_education']))
    selectdata[-1].append(item['tertiary_education'][0]['honour_of_tertiary_education'])
    selectdata[-1].append(item['tertiary_education'][0]['major_of_tertiary_education'])
    selectdata[-1].append(item['tertiary_education'][0]['qualification_of_tertiary_education'])
    selectdata[-1].append(item['tertiary_education'][0]['study_mode_of_tertiary_education'])
    selectdata[-1].append(item['tertiary_education'][0]['university_of_tertiary_education'])
    selectdata[-1].append(int(item['tertiary_education'][0]['tertiary_education_duration_months']) + 12 * int(item['tertiary_education'][0]['tertiary_education_duration_years']))
    selectdata[-1].append(len(item['working_exp']))
    try:
        selectdata[-1].append(sum([int(exp['duration_months']) + int(exp['duration_years']) * 12 for exp in item['working_exp']]))
    except:
        (selectdata[-1].append(0))
header = ['id', 'status', 'age', 'activities num', 'language num', 'scholarship', 'secondary_education num', 
          'tertiary_education num', 'honour', 'major_of_tertiary_education', 'qualification_of_tertiary_education', 
          'study_mode_of_tertiary_education', 'university_of_tertiary_education', 'study time', 'working_exp num', 
          'working_exp total duration_months']
fobj = open('01 select_data_all.csv', 'wb')
[(fobj.write(item), fobj.write(',')) for item in header]
fobj.write('\n')
[([(fobj.write(str(it).replace(',', ' ')), fobj.write(',')) for it in item], fobj.write('\n')) for item in selectdata]
fobj.close()

numdata = copy.deepcopy(selectdata)
# consider it as a binary classification
for i in range(len(numdata) - 1, -1, -1):
    if (numdata[i][1].strip() != 'invited' and numdata[i][1].strip() != 'rejected'):
        del numdata[i]
# all possible values for status
statuslist = list(set([item[1].strip() for item in numdata]))
statusdict = {}
for i in range(len(statuslist)):
    statusdict[statuslist[i]] = i
# all possible values for honour
honourlist = list(set([item[8].strip() for item in numdata]))
honourdict = {}
for i in range(len(honourlist)):
    honourdict[honourlist[i]] = i
# all possible values for major_of_tertiary_education
majorlist = list(set([item[9].strip() for item in numdata]))
majordict = {}
for i in range(len(majorlist)):
    majordict[majorlist[i]] = i
# all possible values for qualification_of_tertiary_education
qualilist = list(set([item[10].strip() for item in numdata]))
qualidict = {}
for i in range(len(qualilist)):
    qualidict[qualilist[i]] = i
# all possible values for study_mode_of_tertiary_education
modelist = list(set([item[11].strip() for item in numdata]))
modedict = {}
for i in range(len(modelist)):
    modedict[modelist[i]] = i
# all possible values for university_of_tertiary_education
unilist = list(set([item[12].strip() for item in numdata]))
unidict = {}

# special treatment on university
for i in range(len(unilist)):
    unidict[unilist[i]] = i
for i in range(len(numdata)):
    numdata[i][1] = statusdict[numdata[i][1].strip()]
    numdata[i][8] = honourdict[numdata[i][8].strip()]
    numdata[i][9] = majordict[numdata[i][9].strip()]
    numdata[i][10] = qualidict[numdata[i][10].strip()]
    numdata[i][11] = modedict[numdata[i][11].strip()]
    numdata[i][12] = unidict[numdata[i][12].strip()]
fobj = open('02 select_data_num.csv', 'wb')
[(fobj.write(item), fobj.write(',')) for item in header]
fobj.write('\n')
[([(fobj.write(str(it).replace(',', ' ')), fobj.write(',')) for it in item], fobj.write('\n')) for item in numdata]
fobj.close()


# artificial neural network
net = buildNetwork(14, 14, 1, bias = True, outclass=SoftmaxLayer)
ds = ClassificationDataSet(14, 1, nb_classes = 2)
for item in numdata:
    ds.addSample(tuple(item[2 : ]), (item[1]))
dsTrain,dsTest = ds.splitWithProportion(0.8)


print('Trainging')
trainer = BackpropTrainer(net, ds, verbose = True, learningrate=0.0001)
# trainer.train()
trainer.trainUntilConvergence(maxEpochs = 20)  
print('Finish training')

Traininp = dsTrain['input']
Traintar = dsTrain['target']
Testinp = dsTest['input']
Testtar = dsTest['target']
forecastTrain = net.activateOnDataset(dsTrain)
print('The accuracy on Training set: ' + str(1 - sum(abs(np.array(forecastTrain) - Traintar)) / float(len(dsTrain))))
forecastTest = net.activateOnDataset(dsTest)
print('The accuracy on Testing set: ' + str(1 - sum(abs(np.array(forecastTest) - Testtar)) / float(len(dsTest))))

headernew = ['forecast_status', 'real_status', 'age', 'activities num', 'language num', 'scholarship', 'secondary_education num', 
          'tertiary_education num', 'honour', 'major_of_tertiary_education', 'qualification_of_tertiary_education', 
          'study_mode_of_tertiary_education', 'university_of_tertiary_education', 'study time', 'working_exp num', 
          'working_exp total duration_months']

Traindata = np.hstack((np.array(Traintar), np.array(Traininp)))
Testdata = np.hstack((np.array(Testtar), np.array(Testinp)))
foreTraindata = np.hstack((forecastTrain, Traindata))
foreTestdata = np.hstack((forecastTest, Testdata))
fobj = open('03 Train_data_num.csv', 'wb')
[(fobj.write(item), fobj.write(',')) for item in headernew]
fobj.write('\n')
[([(fobj.write(str(it).replace(',', ' ')), fobj.write(',')) for it in item], fobj.write('\n')) for item in foreTraindata]
fobj.close()

fobj = open('04 Test_data_num.csv', 'wb')
[(fobj.write(item), fobj.write(',')) for item in headernew]
fobj.write('\n')
[([(fobj.write(str(it).replace(',', ' ')), fobj.write(',')) for it in item], fobj.write('\n')) for item in foreTestdata]
fobj.close()


