from tkinter import *
from PIL import Image as Img, ImageTk
import datetime
import cv2
import os
import numpy as np
import math 
import time
from keras.models import load_model

screen_width_home = 720
screen_height_home = 480

screen_width = 645
screen_height = 620

class GUI(Tk):
    def __init__(self):
        Tk.__init__(self)
        container = Frame(self)
        self.title("SOCIAL DISTANCING AND FACE MASK MONITOR")
        self.iconbitmap(r'CCET_logo3.ico')
        
        self.current_frame="gui"
        print(self.current_frame)

        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
    

        for F,geometry in zip((StartPage, PageOne), (f"{screen_width_home}x{screen_height_home}", f"{screen_width}x{screen_height}", f"{screen_width}x{screen_height}", f"{screen_width}x670")):
            page_name = F.__name__

            frame = F(container, self)

            self.frames[page_name] = (frame, geometry)

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame, geometry = self.frames[page_name]

        # change geometry of the window
        self.update_idletasks()
        self.geometry(geometry)
        frame.tkraise()


    def status(self, value):
        self.statusVar = StringVar()
        self.statusVar.set(value)
        self.statusBar = Label(self, textvariable=self.statusVar, anchor="w", relief=SUNKEN)
        self.statusBar.pack(side=BOTTOM, fill=X)
        
    def video_loop(self):
        # Get frame from the video stream and show it in Tkinter
        ret,image=self.vs.read()  # Read frame by Frame
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  # Converting image into gray
        faces=face_clsfier.detectMultiScale(gray,1.3,5)  # Detect every Faces in Image
        for x,y,w,h in faces:  # Process every Face in Image
        
            face_img=gray[y:y+w,x:x+w]  # Croping the region of interest
            resized=cv2.resize(face_img,(100,100))  # Resize into 100X100
            normalized=resized/255.0  # Normalising the Images by converting pixel range in 0 and 1
            reshaped=np.reshape(normalized,(1,100,100,1))  # Reshaping because NN requires 4D
            result=model.predict(reshaped)  # Predict
    
            label_face=np.argmax(result,axis=1)[0]
            cv2.rectangle(image,(x,y),(x+w,y+h),color_dict[label_face],2)  # Create Bounding R
            cv2.rectangle(image,(x,y-40),(x+w,y),color_dict[label_face],-1)  # Closed rectangle top of the bounding rectangle
            cv2.putText(image, labels_dict[label_face], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) # Show Label
        
        (H, W) = image.shape[:2]
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        
        print("Frame Prediction Time : {:.6f} seconds".format(end - start))
        
        boxes = []
        confidences = []
        classIDs = []
        
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.1 and classID == 0:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
        ind = []
        for i in range(0,len(classIDs)):
            if(classIDs[i]==0):
                ind.append(i)
        a = []
        b = []
    
        if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    a.append(x)
                    b.append(y)
                    
        distance=[] 
        nsd = []
        for i in range(0,len(a)-1):
            for k in range(1,len(a)):
                if(k==i):
                    break
                else:
                    x_dist = (a[k] - a[i])
                    y_dist = (b[k] - b[i])
                    d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                    distance.append(d)
                    if(d <=100):
                        nsd.append(i)
                        nsd.append(k)
                    nsd = list(dict.fromkeys(nsd))
                    print(nsd)
        color = (0, 0, 255) 
        for i in nsd:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            text = "UNSAFE"
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)   
            
        color = (0, 255, 0) 
        if len(idxs) > 0:
            for i in idxs.flatten():
                if (i in nsd):
                    break
                else:
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                    text = 'SAFE'
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)  

        if ret:  # frame captured without any errors
            cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            
            self.current_image = Img.fromarray(cv2image)  # convert image for PIL
            
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            
            self.panel.config(image=imgtk)  # show the image
        
        self.after(30, self.video_loop)  # call the same function after 30 milliseconds

    def take_snapshot(self):
        """ Take snapshot and save it to the file """
        ts = datetime.datetime.now() # grab the current timestamp
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))  # construct filename
        p = os.path.join(self.output_path, filename)  # construct output path
        self.current_image.save(p, "PNG")  # save image as PNG file
        print("[INFO] saved {}".format(filename))

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        window.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

class StartPage(Frame, GUI):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller
        
        ##
        self.current_frame="start"
        print("func start")
        print(self.current_frame)
        ##
        
        backgroundImage = Img.open("Background12_2.jpg")
        self.reset_backgroundImage = ImageTk.PhotoImage(backgroundImage)

        backgroundLabel = Label(self, image=self.reset_backgroundImage)
        backgroundLabel.place(x=0, y=0, relwidth=1, relheight=1)
        backgroundLabel.image = backgroundImage # reference to the image, otherwise the image will be destroyed by the garbage collector when the function returns. 
                                                # Adding a reference as an attribute of the label object.

        backgroundImage2 = PhotoImage(file="Background4.png")        
                 
        head_image = Img.open("head4.png")
        self.reset_head_image = ImageTk.PhotoImage(head_image)
        label = Label(self, image=self.reset_head_image)
        label.pack(side=TOP, fill=X)
    
        
        image3 = Img.open("social-distancing-face-mask-monitor.png")
        self.reset_img3 = ImageTk.PhotoImage(image3)
        self.button3=Button(self,image=self.reset_img3, command=lambda: controller.show_frame("PageOne"))
        self.button3.place(anchor="n", relx=0.5, rely=0.5)

        self.status("HOME PAGE")
      

class PageOne(Frame, GUI):
    def __init__(self, parent, controller, output_path = "./"):
        Frame.__init__(self, parent)
        self.controller = controller
        
        #
        self.current_frame="three"
        print(self.current_frame)
        #
        
        backgroundImage = PhotoImage(file="Background8.png")        
         
        backgroundLabel = Label(self, image=backgroundImage)
        backgroundLabel.place(x=0, y=0, relwidth=1, relheight=1)
        backgroundLabel.image = backgroundImage # reference to the image, otherwise the image will be destroyed by the garbage collector when the function returns. 
                                                # Adding a reference as an attribute of the label object.

        # label = Label(self, text="SOCIAL DISTANCING AND FACE MASK MONITOR", font=LARGE_FONT)
        # label.pack(pady=10, padx=10)
        
        head_image = Img.open("social_distancing_face_mask_monitor2.png")
        self.reset_head_image = ImageTk.PhotoImage(head_image)
        label = Label(self, image=self.reset_head_image, relief="groove")
        label.pack(side=TOP, fill=X)
        
        # Initialize application which uses OpenCV + Tkinter. It displays
        #    a video stream in a Tkinter window and stores current snapshot on disk 
        self.vs = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera
        self.output_path = output_path  # store output path
        self.current_image = None  # current image from the camera

        self.panel = Label(self)  # initialize image panel
        self.panel.pack()
        
        
        image0 = Img.open("camera_button7.png")
        image0 = image0.resize((45,45), Img.ANTIALIAS)
        self.reset_img0 = ImageTk.PhotoImage(image0)
        self.button1=Button(self,image=self.reset_img0, command=self.take_snapshot)
        self.button1.place( relx=0.5, rely=0.85)
        
        image1 = Img.open("home_button10.png")
        image1 = image1.resize((45,45), Img.ANTIALIAS)
        self.reset_img1 = ImageTk.PhotoImage(image1)
        self.button1=Button(self,image=self.reset_img1, command=lambda: controller.show_frame("StartPage"))
        self.button1.place(relx=0, rely=0.19)
            
        image4 = Img.open("close_button1.png")
        image4 = image4.resize((45,45), Img.ANTIALIAS)
        self.reset_img4 = ImageTk.PhotoImage(image4)
        self.button4=Button(self,image=self.reset_img4, command=self.destructor)
        self.button4.place( relx=0.92, rely=0.19)
        
        self.status("SOCIAL DISTANCING AND FACE MASK MONITOR")
        self.video_loop()


if __name__ == "__main__":
        
    model = load_model('model-020.model')  # Loading the Model
    
    face_clsfier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Casscade classifier to get the region of interest

    labels_dict={0:'WITH MASK',1:'WITHOUT MASK'}  # Creating dictionary in which 0 : WITH MASK and 1 : WITHOUT MASK
    color_dict={0:(0,255,0),1:(0,0,255)}   # Creating dictionary in which 0 : GREEN COLOR and 1 : RED COLOR
    
    labelsPath = "./coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    
    weightsPath = "./yolov3.weights"
    configPath = "./yolov3.cfg"
    
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    window = GUI()
    window.mainloop()

