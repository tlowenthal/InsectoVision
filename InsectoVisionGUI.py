import os
from shutil import rmtree
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import inference_pipeline

#Constants
x_max = 1800
y_max = 900
n_img = 100
img_path = "MiniSample"
bbox_path = "output"
save_dir = "guiout"
DEFAULT_LABEL = "undefined"
conf_threshold = 0.85


current = 0
selected = []
entoboxes = []
drawn_bboxes = []
img_id = None
classes = []


drawing = 0                #O is not currently drawing, 1 if drawing initialized (draw box button pressed), 2 if one point of box drawn
drawing_reason = 0
NEW_BBOX = 1
SELECTING = 2
draw_coord = None
draw_indic = None


SURE=1
DOUBT=2
CONFIRMED=3
REJECTED=4
SELECTED=5
COLORS = {SURE:"chartreuse4",DOUBT:"gold",CONFIRMED:"green2",REJECTED:"red",SELECTED:"blue"}


BWIDTH = 15 #Button width
PADX = 5 #x axis padding between buttons/labels

class EntoBox:

    to_add = []


    def __init__(self,name):
        self.name = name
        print("Loading "+name)

        #Get the image
        img = Image.open(os.path.join(img_path,name+".jpg"))
        self.dim = get_dim((img.size))
        self.image = ImageTk.PhotoImage(img.resize(self.dim))
        
        #Get the bboxes
        self.bboxes = []

        if(os.path.isfile(os.path.join(bbox_path,name+".txt"))):
            txt = open(os.path.join(bbox_path,name+".txt"))

            #Compute bbox coordinates from yolo notation
            for line in txt:
                la = line.split(" ")[0:6]                                           #TODO mettre un normalize a la fin du net pour retirer ce truc
                [x,y,w,h] = [float(la[1]),float(la[2]),float(la[3]),float(la[4])]
                x1 = int((x-w/2)*self.dim[0])
                x2 = int((x+w/2)*self.dim[0])
                y1 = int((y-h/2)*self.dim[1])
                y2 = int((y+h/2)*self.dim[1])
                if(len(la) == 6):
                    c = float(la[5])
                else:
                    c = 1
                self.bboxes.append(BBox([x1,y1,x2,y2],c,self))
            txt.close()
    
    def show(self):
        global img_id
        global selected
        selected = []
        img_id = canvas.create_image(int(self.dim[0]/2),int(self.dim[1]/2),image=entoboxes[current].image,tags=["picture"])

        for bbox in self.bboxes:
            bbox.draw()


class BBox:

    status = None
    itemId = None
    label = DEFAULT_LABEL

    def __init__(self,coord,conf,parent):
        self.parent = parent
        self.coord = coord
        self.conf = conf
        if(conf < conf_threshold):
            self.status = DOUBT
        else:
            self.status = SURE
    
    def draw(self):
        global canvas
        global drawn_bboxes        
        boxid = canvas.create_rectangle(self.coord[0],self.coord[1],self.coord[2],self.coord[3],outline=COLORS[self.status],width=2,tags=["bbox"])
        self.itemId = boxid
        drawn_bboxes.append(self)
    
    def redraw(self):
        global canvas
        canvas.delete(self.itemId)
        self.draw()

    def to_yolo(self):
        [d0,d1] = self.parent.dim
        [x1,y1,x2,y2] = self.coord
        x = ((float(x2+x1))/2)/d0
        y = ((float(y2+y1))/2)/d1
        w = float(abs(x2-x1))/d0
        h = float(abs(y2-y1))/d1

        return [x,y,w,h]



def get_dim(dim):
    x = dim[0]
    y = dim[1]
    scale = min(x_max/x,y_max/y)
    return(int(x*scale),int(y*scale))

def on_click(e):  #Drawing boxes is in here
    global canvas
    global draw_coord
    global drawing
    global draw_indic
    global drawing_reason

    if drawing == 0:
        select(e)
    elif drawing == 1:
        draw_coord = [e.x,e.y]
        draw_indic = canvas.create_oval(draw_coord[0],draw_coord[1],draw_coord[0],draw_coord[1],fill=COLORS[SELECTED],outline=COLORS[SELECTED],width=4)
        drawing = 2
    elif drawing == 2:
        canvas.delete(draw_indic)
        x1 = min(draw_coord[0],e.x)
        x2 = max(draw_coord[0],e.x)
        y1 = min(draw_coord[1],e.y)
        y2 = max(draw_coord[1],e.y)
        if drawing_reason == NEW_BBOX:
            new = BBox([x1,y1,x2,y2],1,entoboxes[current])
            new.status = CONFIRMED
            entoboxes[current].bboxes.append(new)
            new.draw()
            update_count()
        elif drawing_reason == SELECTING:
            unselect()
            for bbox in entoboxes[current].bboxes:
                c = bbox.coord
                if(x1<c[0] and y1<c[1] and x2>c[2] and y2>c[3]):
                    selected.append(bbox)
                    canvas.itemconfig(bbox.itemId,outline = COLORS[SELECTED])
        drawing = 0
        drawing_reason = 0


def select(e,cumul = False):
    global selected

    found = None

    for bbox in entoboxes[current].bboxes:
        c = bbox.coord 
        if (e.x>c[0] and e.y>c[1] and e.x<c[2] and e.y < c[3] and bbox not in selected):
            canvas.itemconfig(bbox.itemId,outline = COLORS[SELECTED])
            found = bbox
            break

    if not cumul:
        unselect()

    if found != None:
        if cumul:
            selected.append(found)
        else:
            selected = [found]

def select_many(e):
    select(e,True)
def select_group():
    start_draw(reason=SELECTING)


#Button functions
def next():
    change_img(1)
def prev():
    change_img(-1)
def change_img(n):
    global current
    global title_label
    global number_label
    global save_label

    save_label.config(text="")
    current = (current+n)%n_img
    title_label.config(text="Image "+str(current+1)+" /"+str(n_img))
    canvas.delete(img_id)
    for bbox in drawn_bboxes:
        canvas.delete(bbox.itemId)
    entoboxes[current].show()
    update_count()
def unselect():
    global selected
    for bbox in selected:
        bbox.redraw()
    selected = []

def rate_g():
    for box in selected:
        box.status = CONFIRMED
        box.redraw()
    unselect()
    update_count()
def rate_b():
    for box in selected:
        box.status = REJECTED
        box.redraw()
    unselect()
    update_count()
def save():
    global save_label
    missing_tag = False
    for bbox in entoboxes[current].bboxes:
        if bbox.status == DOUBT:
            missing_tag = True
            break
    if missing_tag:
        save_label.config(text="Save Failed, Unresolved Tags")
        return
    
    if classes == []:
        get_classes()
    
    lf = open(os.path.join(save_dir,"classes.txt"),"a")
    f = open(os.path.join(save_dir,entoboxes[current].name)+".txt","w")

    for bbox in entoboxes[current].bboxes:
        if bbox.status != REJECTED:
            if bbox.label not in classes:
                lf.write(bbox.label+"\n")
                classes.append(bbox.label)
            cnum = classes.index(bbox.label)
            f.write(str(cnum)+" "+" ".join(str(x) for x in bbox.to_yolo())+ "\n")

    f.close()
    lf.close()

    save_label.config(text="Save Successful")

def get_classes():
    global classes
    name = os.path.join(save_dir,"classes.txt")
    if not os.path.isfile(name):
        return
    else:
        lf = open(name,"r")
        classes = []
        for line in lf:
            classes.append(line)
        lf.close()
        
def combine():
    global selected
    if len(selected)<2:
        return
    coord = selected[0].coord.copy()
    for bbox in selected:
        if bbox.coord[0] < coord[0]:
            coord[0] = bbox.coord[0]
        if bbox.coord[1] < coord[1]:
            coord[1] = bbox.coord[1]
        if bbox.coord[2] > coord[2]:
            coord[2] = bbox.coord[2]
        if bbox.coord[3] > coord[3]:
            coord[3] = bbox.coord[3]
    
    entoboxes[current].bboxes = [box for box in entoboxes[current].bboxes if box not in selected]
    for bbox in selected:
        canvas.delete(bbox.itemId)
    selected = []
    new = BBox(coord,1,entoboxes[current])
    new.status = CONFIRMED
    new.draw()
    entoboxes[current].bboxes.append(new)    
    unselect()
    update_count()
def add_label():
    

    label_window = Toplevel()
    label_window.config(width=600,height=100)
    label_window.geometry('+500+500')
    tfrm = ttk.Frame(label_window, padding=5)
    tfrm.grid()
    ttk.Label(tfrm,text="Enter label name").grid(row=0,column=0)
    e = ttk.Entry(tfrm)
    e.grid(row=1,column=0)
    e.focus()

    def conf_label():
        for bbox in selected:
            bbox.label = e.get()
        label_window.destroy()

    ttk.Button(tfrm,text="Ok",command=conf_label).grid(row=2,column=0)


def update_thresh(val):
    global conf_threshold 
    global thresh_label
    val = int(float(val))
    conf_threshold = float(val)/100
    thresh_label.config(text= "Confidence threshold: "+ str(val)+"%")
    for bbox in entoboxes[current].bboxes:
        if bbox.status == DOUBT and bbox.conf > conf_threshold:
            bbox.status = SURE
        elif bbox.status == SURE and bbox.conf < conf_threshold:
            bbox.status = DOUBT 
    change_img(0) #Redraws current entobox 
    update_count()
def update_count():
    global number_label
    cnt = 0
    for bbox in entoboxes[current].bboxes:
        if bbox.status == CONFIRMED or bbox.status == SURE:
            cnt += 1
    number_label.config(text= str(cnt)+" speciments detected")

def start_draw(reason = NEW_BBOX):
    global drawing
    global drawing_reason
    drawing_reason = reason
    drawing = 1



#if os.path.exists("output"):
    #rmtree("output")

#inference_pipeline.main(img_path,write_conf=True)

#Setup the window
root = Tk()
root.title("InsectoVision")
frm = ttk.Frame(root, padding=10)
frm.grid()

#Get images
for entry in os.listdir(img_path):
    if(entry.endswith(".jpg")):
        entoboxes.append(EntoBox(entry[:len(entry)-4]))
n_img = len(entoboxes)



#Title and buttons
title_label = ttk.Label(frm, text="Image "+str(current+1)+" /"+str(n_img))
title_label.grid(column=1, row=0)
number_label = ttk.Label(frm, text= str(len(entoboxes[current].bboxes))+" speciments detected",width=23)
number_label.grid(column=2,row=0)
#save_label = ttk.Label(frm,text="Unsaved")
ttk.Button(frm, text="Next", command=next,width=BWIDTH).grid(column=2, row=1,padx=PADX)
ttk.Button(frm, text="Previous", command=prev,width=BWIDTH).grid(column=1, row=1,padx=PADX)
ttk.Button(frm,text="Good detection",command=rate_g,width=BWIDTH).grid(column=5,row=0,padx=PADX)
ttk.Button(frm,text="Bad detection",command=rate_b,width=BWIDTH).grid(column=5,row=1,padx=PADX)
ttk.Button(frm,text="Combine boxes",command=combine,width=BWIDTH).grid(column=6,row=1,padx=PADX)
ttk.Button(frm,text="New box",command=start_draw,width=BWIDTH).grid(column=6,row=0,padx=PADX)
ttk.Button(frm,text="Add label",command=add_label,width=BWIDTH).grid(column=7,row=0,padx=PADX)
ttk.Button(frm,text="Select Group",command=select_group,width=BWIDTH).grid(column=7,row=1,padx=PADX)

ttk.Button(frm,text="Save",command=save,width=BWIDTH).grid(column=9,row=1,padx=PADX)
save_label = ttk.Label(frm)
save_label.grid(column=9,row=0,padx=PADX)

#Canvas for bounding boxes
canvas = Canvas(root,width=x_max,height=y_max)
canvas.grid(column=0,row=2,padx=20)
canvas.bind('<Button-1>',on_click)
canvas.bind('<Shift-Button-1>',select_many)
entoboxes[0].show()

thresh_label = ttk.Label(frm,width=26)
thresh_label.grid(column=8,row=0,padx=PADX)

thresh_scale = ttk.Scale(frm, from_=65,to=100,command=update_thresh)
thresh_scale.grid(column=8,row=1,padx=PADX)
thresh_scale.set(100*conf_threshold)



root.mainloop()