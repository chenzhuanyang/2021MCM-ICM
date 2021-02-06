import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np,pandas as pd
import cv2,os,time,random,zipfile,fitz,re,shutil
my_Data1="C:/Users/ASUS/Desktop/problem c/mydata.xlsx"
mydata=pd.read_excel(my_Data1,engine='openpyxl')
da=mydata[['GlobalID','Lab Status','FileName']]
DATA=[]

length=len(da['GlobalID'])
for i in range(length):
    temp=da.loc[i,'Lab Status']
    if temp=='Positive ID' or temp=='Unprocessed' or temp=='Unverified':
        if da.loc[i,'FileName'] :
            temp=[]
            temp.append(da.loc[i,'GlobalID'])
            temp.append(da.loc[i,'Lab Status'])
            temp.append(da.loc[i,'FileName'])
            DATA.append(temp)
pd_DATA=pd.DataFrame(DATA,columns=['GlobalID','Lab Status','FileName'])
model=load_model("C:/Users/ASUS/Desktop/problem c/model03_59_0.02.h5")
IMG_SIZE = 64
path="C:/Users/ASUS/Desktop/problem c/2021MCM_ProblemC_Files"

Positive_ID_Probability=[]
for FileName in  pd_DATA['FileName'].values.tolist():
    this_path=os.path.join(path,FileName)
    if ('png' in FileName or 'jpg'in FileName or 'jfif'in FileName):
        create_data_from_img(this_path,Positive_ID_Probability)
    elif ('pdf'in FileName):
        create_data_from_pdf(this_path,Positive_ID_Probability)
    elif('docx'in FileName):
        create_data_from_docx(this_path,Positive_ID_Probability)
        
    elif('MOV'in FileName or 'mp4'in FileName):
        create_data_from_video(this_path,Positive_ID_Probability)
    elif('zip'in FileName):
        create_data_from_zip(this_path,Positive_ID_Probability)
    else:
        Positive_ID_Probability.append("Error!")
        
pd_DATA['Positive ID Probability']=Positive_ID_Probability
writer=pd.ExcelWriter('C:/Users/ASUS/Desktop/problem c/mydata4.xlsx')
pd_DATA.to_excel(writer,index=False)
writer.save()

def predict_from_img(this_path,img=[]):
    if img==[]:
        img_array= cv2.imread(this_path,1)
    else:
        img_array=img
    image = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prob = model.predict(image)[0]
    result=prob[0]*100
    return result
def create_data_from_img(this_path,Positive_ID_Probability):
    Positive_ID_Probability.append(predict_from_img(this_path))
def create_data_from_pdf(this_path,Positive_ID_Probability):
    pic_path = this_path[0:len(this_path)-4] 
    if os.path.exists(pic_path):
        pass
    else:
        os.mkdir(pic_path)
    t0 = time.clock()                          
    checkXO = r"/Type(?= */XObject)"           
    checkIM = r"/Subtype(?= */Image)"
    doc = fitz.open(this_path)                      
    imgcount = 0                               
    lenXREF = doc._getXrefLength()             
    result=0
    
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

        result=(result+predict_from_img(os.path.join(pic_path, new_name)))/2
    Positive_ID_Probability.append(result)

def create_data_from_docx(this_path,Positive_ID_Probability):
    path=this_path
    zip_path=this_path[0:len(this_path)-4]+'zip'
    tmp_path=this_path[0:len(this_path)-5]+'temp'
    store_path=this_path[0:len(this_path)-5]

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

    result=0

    for i in pic:
        new_name = path.replace('\\', '_')
        new_name = new_name.replace(':', '') + '_' + i
        shutil.copy(os.path.join(tmp_path + '/word/media', i), os.path.join(store_path, new_name))

        result=(result+predict_from_img(os.path.join(store_path, new_name)))/2
    Positive_ID_Probability.append(result)

    for i in os.listdir(tmp_path):
        if os.path.isdir(os.path.join(tmp_path, i)):
            shutil.rmtree(os.path.join(tmp_path, i))

def create_data_from_video(this_path,Positive_ID_Probability):
    cap = cv2.VideoCapture(this_path)
    result=0
    
    while cap.isOpened():
        rval, image = cap.read()
        if rval==True:
            result =(result+predict_from_img(this_path,image))/2
        else:
            break
    Positive_ID_Probability.append(result)
def create_data_from_zip(this_path,Positive_ID_Probability):
    result=0
    with zipfile.ZipFile(this_path, mode='r') as zfile: 
        for name in zfile.namelist():  
            if ('.jpg' not in name) or ('.JPG'not in name):
                continue
            with zfile.open(name,mode='r') as image_file:
                content = image_file.read() 
                image = np.asarray(bytearray(content), dtype='uint8')
                result=(result+predict_from_img(this_path,image))/2
    zfile.close()
    Positive_ID_Probability.append(result)