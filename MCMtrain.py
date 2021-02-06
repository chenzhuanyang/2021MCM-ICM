import matplotlib.pyplot as  plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os,cv2,random,zipfile,fitz,re,shutil,time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
tf.__version__

EXCEL_Data1="C:/Users/ASUS/Desktop/problem c/2021_MCM_Problem_C_Data/ \
        2021MCMProblemC_DataSet.xlsx"
EXCEL_Data2="./2021_MCM_Problem_C_Data/2021MCM_ProblemC_ Images_by_GlobalID.xlsx"
AllDataOfExcel1=pd.read_excel(EXCEL_Data1,engine='openpyxl')
AllDataOfExcel2=pd.read_excel(EXCEL_Data2,engine='openpyxl')

da=AllDataOfExcel1[['GlobalID','Lab Status','Latitude','Longitude']]
mylist=AllDataOfExcel2['GlobalID'].values.tolist()

length=len(da['GlobalID'])
temp=[0]*length
for i in range(length):
    if da.loc[i,'GlobalID'] in mylist:
        temp[i]=AllDataOfExcel2.loc[mylist.index(da.loc[i,'GlobalID']),'FileName']
da['FileName']=temp

writer=pd.ExcelWriter('C:/Users/ASUS/Desktop/problem c/mydata.xlsx')
da.to_excel(writer)
writer.save()

DataOfName=[]
DataOfLab=[]
mylist=[]
mylist=da['FileName'].values.tolist()
for i in range(len(mylist)):
    
    if mylist[i]:
        
        if da.loc[i,'Lab Status'] in ['Positive ID', 'Negative ID']:
            DataOfName.append(mylist[i])
            DataOfLab.append(da.loc[i,'Lab Status'])
mydata2=pd.DataFrame([DataOfName,DataOfLab])
mydata2=mydata2.T
mydata2.columns=['FileName','Lab Status']

writer=pd.ExcelWriter('C:/Users/ASUS/Desktop/problem c/mydata3.xlsx')
mydata2.to_excel(writer)
writer.save()

NUM_CLASSES = 2
BATCH_SIZE = 64
Lab_Data_Path = "./2021_MCM_Problem_C_Data"
Img_Data_Path = "./2021MCM_ProblemC_Files"

CATEGORIES = ['Positive ID', 'Negative ID']
image_size = 64
EPOCH=80
NUM_EPOCH=2

Positive_img_data,Positive_lab_data,Negative_img_data,
    Negative_lab_data=create_data(Img_Data_Path,DataOfName)

X_Train_Positive,X_Test_Positive,Y_Train_Positive,Y_Test_Positive=
        train_test_split(Positive_img_data,Positive_lab_data,test_size=0.25)
X_Train_Negative,X_Test_Negative,Y_Train_Negative,Y_Test_Negative=
        train_test_split(Negative_img_data,Negative_lab_data,test_size=0.25)

X_train=X_Train_Positive+X_Train_Negative
Y_train=Y_Train_Positive+Y_Train_Negative
X_test =X_Test_Positive+X_Test_Negative
Y_test =Y_Test_Positive+Y_Test_Negative

TrainData=[]
TestData=[]
for i in range(len(X_train)):
    TrainData.append([X_train[i],Y_train[i]])
for i in range(len(X_test)):
    TestData.append([X_test[i],Y_test[i]])
    
random.shuffle(TrainData)
random.shuffle(TestData)


TrainData = TrainData_process(TrainData)
TestData = TestData_process(TestData)

TotalOfTrain=len(X_train)
TotalOfTest=len(X_test)
print(TotalOfTrain)
print(TotalOfTest)


BATCH_SIZE=64
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', 
                 input_shape=(image_size,image_size,3)))
model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
model.add(Dropout(0.25))       

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 1), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(1, 3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(1, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
model.add(Dropout(0.5))


model.add(Flatten())

model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=2, activation = 'softmax'))


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',metrics=['acc'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model03_{epoch:02d}_{val_loss:.2f}.h5', 
        save_best_only=True,monitor='val_loss',verbose=1                    
    )
]

history = model.fit_generator(
    TrainData,
    steps_per_epoch=TotalOfTrain // BATCH_SIZE,    
    epochs= EPOCH,                                   
    validation_data=TestData,                 
    validation_steps=TotalOfTest // BATCH_SIZE,     
    callbacks = callbacks                         
)

PlotOfModel2(history)
model.summary()

######## Some function ##########################
def PlotOfModel2(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
def TrainData_process(data):
    X = []
    Y = []

    for features,label in data:
        X.append(features)
        Y.append(label)

    X = np.array(X).reshape(-1, image_size, image_size, 3)
    Y = np.array(Y)
    
    train_image_generator = ImageDataGenerator(
        rescale=1./255,         
        rotation_range=45,      
        horizontal_flip=True,   
        vertical_flip=True,     
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        zoom_range=0.5          
        )
    train_data_gen = train_image_generator.flow(X, Y, batch_size = BATCH_SIZE)  
    return train_data_gen

def TestData_process(data):
    X = []
    Y = []

    for features,label in data:
        X.append(features)
        Y.append(label)

    X = np.array(X).reshape(-1, image_size, image_size, 3)
    Y = np.array(Y)    
    validation_image_generator = ImageDataGenerator(rescale=1./255)
    val_data_gen = validation_image_generator.flow(X, Y, batch_size = BATCH_SIZE) 
    return val_data_gen

def create_data(DIR,DataOfName):
    data=[]
    Positive_img_data=[]
    Positive_lab_data=[]
    Negative_img_data=[]
    Negative_lab_data=[]
    for i in range(len(DataOfName)):
        FileName=DataOfName[i]
        path=os.path.join(DIR,FileName)
        if ('png' in FileName or 'jpg'in FileName or 'jfif'in FileName):
            create_data_from_img(path,Positive_img_data,Positive_lab_data,
                                 Negative_img_data,Negative_lab_data,DataOfLab[i])
        elif ('pdf'in FileName):
            create_data_from_pdf(path,Positive_img_data,Positive_lab_data,
                                 Negative_img_data,Negative_lab_data,DataOfLab[i])
        elif('docx'in FileName):
            create_data_from_docx(path,Positive_img_data,Positive_lab_data,
                                  Negative_img_data,Negative_lab_data,DataOfLab[i])
        elif('MOV'in FileName or 'mp4'in FileName):
            create_data_from_video(path,Positive_img_data,Positive_lab_data,
                                   Negative_img_data,Negative_lab_data,DataOfLab[i])
        elif('zip'in FileName):
            create_data_from_zip(path,Positive_img_data,Positive_lab_data,
                                 Negative_img_data,Negative_lab_data,DataOfLab[i])
    
    return Positive_img_data,Positive_lab_data,Negative_img_data,Negative_lab_data

def append_data(Positive_img_data,Positive_lab_data,Negative_img_data,
                Negative_lab_data,new_array,class_num):
    if class_num==0:
        Positive_img_data.append(new_array)
        Positive_lab_data.append(class_num)
    elif class_num==1:
        Negative_img_data.append(new_array)
        Negative_lab_data.append(class_num)

def create_data_from_img(DIR,Positive_img_data,Positive_lab_data,
                         Negative_img_data,Negative_lab_data,category):
    class_num=CATEGORIES.index(category) 
    img_array= cv2.imread(DIR) 
    new_array= cv2.resize(img_array,(image_size,image_size))

    append_data(Positive_img_data,Positive_lab_data,Negative_img_data,
                Negative_lab_data,new_array,class_num)
    
    
def create_data_from_pdf(DIR,Positive_img_data,Positive_lab_data,
                         Negative_img_data,Negative_lab_data,category):
    class_num=CATEGORIES.index(category)
    pic_path = DIR[0:len(DIR)-4] 
    if os.path.exists(pic_path):
        pass
    else:
        os.mkdir(pic_path)

    t0 = time.clock()                          
    checkXO = r"/Type(?= */XObject)"           
    checkIM = r"/Subtype(?= */Image)"
    doc = fitz.open(DIR)                      
    imgcount = 0                               
    lenXREF = doc._getXrefLength()             
     
    for i in range(1, lenXREF):
        text = doc._getXrefString(i)            
        isXObject = re.search(checkXO, text)    
        isImage = re.search(checkIM, text)      
        if not isXObject or not isImage:        
            continue
        imgcount += 1

        pix = fitz.Pixmap(doc, i)               
        new_name = "Img{}.png".format(imgcount) 
        if pix.n < 5:                           
            pix.writePNG(os.path.join(pic_path, new_name))
        else:                             
            pix0 = fitz.Pixmap(fitz.csRGB, pix)
            pix0.writePNG(os.path.join(pic_path, new_name))
            pix0 = None
        pix = None                              
        t1 = time.clock()                       
        print("run:{}s".format(t1 - t0))
        print("Total take {} imges".format(imgcount))

        img_array= cv2.imread(os.path.join(pic_path, new_name)) 
        new_array= cv2.resize(img_array,(image_size,image_size))

        append_data(Positive_img_data,Positive_lab_data,Negative_img_data,
                    Negative_lab_data,new_array,class_num)
        
def create_data_from_docx(DIR,Positive_img_data,Positive_lab_data,
                          Negative_img_data,Negative_lab_data,category):
    class_num=CATEGORIES.index(category) 
    path=DIR
    zip_path=DIR[0:len(DIR)-4]+'zip'
    tmp_path=DIR[0:len(DIR)-5]+'temp'
    store_path=DIR[0:len(DIR)-5]

    if os.path.exists(tmp_path):
        pass
    else:
        os.mkdir(tmp_path)
    if os.path.exists(store_path):
        pass
    else:
        os.mkdir(store_path)
    
    os.rename(path, zip_path)
    f = zipfile.ZipFile(zip_path, 'r')
    for file in f.namelist():
        f.extract(file, tmp_path)
    f.close()

    os.rename(zip_path, path)
    pic = os.listdir(os.path.join(tmp_path, 'word/media'))

    for i in pic:
        new_name = path.replace('\\', '_')
        new_name = new_name.replace(':', '') + '_' + i
        shutil.copy(os.path.join(tmp_path + '/word/media', i), 
                    os.path.join(store_path, new_name))

        img_array= cv2.imread(os.path.join(store_path, new_name)) 
        new_array= cv2.resize(img_array,(image_size,image_size))

        append_data(Positive_img_data,Positive_lab_data,
                    Negative_img_data,Negative_lab_data,new_array,class_num)

    for i in os.listdir(tmp_path):
        if os.path.isdir(os.path.join(tmp_path, i)):
            shutil.rmtree(os.path.join(tmp_path, i))

def create_data_from_video(DIR,Positive_img_data,Positive_lab_data,
                           Negative_img_data,Negative_lab_data,category):
    cap = cv2.VideoCapture(DIR)
    class_num=CATEGORIES.index(category) 
    count=0
    count_max=24-1
    while cap.isOpened():
        rval, image = cap.read()
        if rval==True:
            if count==count_max:
                new_array = cv2.resize(image, (image_size, image_size))  
                append_data(Positive_img_data,Positive_lab_data,
                            Negative_img_data,Negative_lab_data,new_array,class_num)
                count=0
            else:
                count+=1
        else:
            break
    
def create_data_from_zip(DIR,Positive_img_data,Positive_lab_data,
                         Negative_img_data,Negative_lab_data,category):
    class_num=CATEGORIES.index(category) 
    with zipfile.ZipFile(DIR, mode='r') as zfile: 
 
        for name in zfile.namelist():  #
            if ('.jpg' not in name) or ('.JPG'not in name):
                continue
            with zfile.open(name,mode='r') as image_file:
                content = image_file.read() 
                image = np.asarray(bytearray(content), dtype='uint8')
                new_array = cv2.resize(image, (image_size, image_size))  
                
                append_data(Positive_img_data,Positive_lab_data,Negative_img_data,
                            Negative_lab_data,new_array,class_num)

    zfile.close()