import os
from shutil import rmtree
from tkinter import *
from tkinter import filedialog as fd
from tkinter import ttk
from PIL import ImageTk, Image
import inference_pipeline

#Constants
DEFAULT_LABEL = "undefined"


NEW_BBOX = 1
SELECTING = 2

SURE=1
DOUBT=2
CONFIRMED=3
REJECTED=4
SELECTED=5
COLORS = {SURE:"chartreuse4",DOUBT:"gold",CONFIRMED:"green2",REJECTED:"red",SELECTED:"blue"}


BWIDTH = 15 #Button width
PADX = 5 #x axis padding between buttons/labels
NONCANVASHEIGHT = 150
NONCANVASWIDTH = 30

class EntoBox:

    to_add = []


    def __init__(self,name,img_path,bbox_path,ct,gui):
        self.name = name
        print("Loading "+name)

        #Get the image
        img = Image.open(os.path.join(img_path,name+".jpg"))
        self.dim = gui.get_dim((img.size))
        print(self.dim)
        self.image = ImageTk.PhotoImage(img.resize(self.dim))
        
        #Get the bboxes
        self.bboxes = []

        if(os.path.isfile(os.path.join(bbox_path,name+".txt"))):
            txt = open(os.path.join(bbox_path,name+".txt"))

            #Compute bbox coordinates from yolo notation
            for line in txt:
                la = line.split(" ")[0:6]                                         
                [x,y,w,h] = [float(la[1]),float(la[2]),float(la[3]),float(la[4])]
                x1 = int((x-w/2)*self.dim[0])
                x2 = int((x+w/2)*self.dim[0])
                y1 = int((y-h/2)*self.dim[1])
                y2 = int((y+h/2)*self.dim[1])
                if(len(la) == 6):
                    c = float(la[5])
                else:
                    c = 1
                self.bboxes.append(BBox([x1,y1,x2,y2],c,ct,self))
            txt.close()
    
    def show(self,gui):
        gui.selected = []
        gui.img_id = gui.canvas.create_image(int(self.dim[0]/2),int(self.dim[1]/2),image=gui.entoboxes[gui.current].image,tags=["picture"])

        for bbox in self.bboxes:
            bbox.draw(gui)


class BBox:

    status = None
    itemId = None
    label = DEFAULT_LABEL

    def __init__(self,coord,conf,ct,parent):
        self.parent = parent
        self.coord = coord
        self.conf = conf
        if(conf < ct):
            self.status = DOUBT
        else:
            self.status = SURE
    
    def draw(self,gui):      
        boxid = gui.canvas.create_rectangle(self.coord[0],self.coord[1],self.coord[2],self.coord[3],outline=COLORS[self.status],width=2,tags=["bbox"])
        self.itemId = boxid
        gui.drawn_bboxes.append(self)
    
    def redraw(self,gui):
        gui.canvas.delete(self.itemId)
        self.draw(gui)

    def to_yolo(self):
        [d0,d1] = self.parent.dim
        [x1,y1,x2,y2] = self.coord
        x = ((float(x2+x1))/2)/d0
        y = ((float(y2+y1))/2)/d1
        w = float(abs(x2-x1))/d0
        h = float(abs(y2-y1))/d1

        return [x,y,w,h]

class GUI:

    started = False
    entoboxes = []
    drawn_bboxes = []
    selected = []
    classes = []
    current = 0
    img_id = None
    img_path = "MiniSample"
    bbox_path = "output"
    save_dir = "guiout"
    n_img = 0
    conf_threshold = 0.85

    drawing = 0                #O is not currently drawing, 1 if drawing initialized (draw box button pressed), 2 if one point of box drawn
    drawing_reason = 0
    draw_coord = None
    draw_indic = None

    def __init__(self):
        root = Tk()
        root.minsize(300,150)
        root.title("InsectoVision")
        frm = ttk.Frame(root, padding=1)
        frm.grid()
        self.y_max = root.winfo_screenheight()-NONCANVASHEIGHT
        self.x_max = root.winfo_screenwidth()-NONCANVASWIDTH
        self.root = root
        self.frame = frm

        menubar = Menu(root)
        root.config(menu=menubar)
        filemenu = Menu(menubar,tearoff=False)
        filemenu.add_command(label="Image directory",command=self.choose_input)
        filemenu.add_command(label="Output directory",command=self.choose_output)
        menubar.add_cascade(label="File",menu=filemenu)

    def choose_input(self):
        self.img_path = fd.askdirectory()
        print("Input directory selected")
        if os.path.exists("output"):
            rmtree("output")
        inference_pipeline.main(self.img_path,write_conf=True)
        self.load_images()
        if(not self.started):
           self.start()

    def choose_output(self):
        self.save_dir = fd.askdirectory()

    def start(self):
        self.make_interface()
        self.make_canvas()
        self.make_thresh()

    def load_images(self):
        for entry in os.listdir(self.img_path):
            if(entry.endswith(".jpg")):
                self.entoboxes.append(EntoBox(entry[:len(entry)-4],self.img_path,self.bbox_path,self.conf_threshold,self))
        self.n_img = len(self.entoboxes)

    def make_interface(self):
        #Title and buttons
        self.title_label = ttk.Label(self.frame, text="Image "+str(self.current+1)+" /"+str(self.n_img))
        self.title_label.grid(column=1, row=0)
        self.number_label = ttk.Label(self.frame, text= str(len(self.entoboxes[self.current].bboxes))+" speciments detected",width=23)
        self.number_label.grid(column=2,row=0)
        ttk.Button(self.frame,text="Next", command=self.next,width=BWIDTH).grid(column=2, row=1,padx=PADX)
        ttk.Button(self.frame,text="Previous", command=self.prev,width=BWIDTH).grid(column=1, row=1,padx=PADX)
        ttk.Button(self.frame,text="Good detection",command=self.rate_g,width=BWIDTH).grid(column=5,row=0,padx=PADX)
        ttk.Button(self.frame,text="Bad detection",command=self.rate_b,width=BWIDTH).grid(column=5,row=1,padx=PADX)
        ttk.Button(self.frame,text="Combine boxes",command=self.combine,width=BWIDTH).grid(column=6,row=1,padx=PADX)
        ttk.Button(self.frame,text="New box",command=self.start_draw,width=BWIDTH).grid(column=6,row=0,padx=PADX)
        ttk.Button(self.frame,text="Add label",command=self.add_label,width=BWIDTH).grid(column=7,row=0,padx=PADX)
        ttk.Button(self.frame,text="Select Group",command=self.select_group,width=BWIDTH).grid(column=7,row=1,padx=PADX)

        ttk.Button(self.frame,text="Save",command=self.save,width=BWIDTH).grid(column=9,row=1,padx=PADX)
        self.save_label = ttk.Label(self.frame)
        self.save_label.grid(column=9,row=0,padx=PADX)

    def make_canvas(self):
        #Canvas for bounding boxes
        self.canvas = Canvas(self.root,height=self.y_max,width=self.x_max)
        self.canvas.grid(column=0,row=2,padx=20)
        self.canvas.bind('<Button-1>',self.on_click)
        self.canvas.bind('<Shift-Button-1>',self.select_many)
        self.entoboxes[0].show(self)

    def make_thresh(self):
        self.thresh_label = ttk.Label(self.frame,width=26)
        self.thresh_label.grid(column=8,row=0,padx=PADX)

        self.thresh_scale = ttk.Scale(self.frame, from_=65,to=100,command=self.update_thresh)
        self.thresh_scale.grid(column=8,row=1,padx=PADX)
        self.thresh_scale.set(100*self.conf_threshold)

    def next(self):
        self.change_img(1)
    def prev(self):
        self.change_img(-1)
    def change_img(self,n):
        self.save_label.config(text="")
        self.current = (self.current+n)%self.n_img
        self.title_label.config(text="Image "+str(self.current+1)+" /"+str(self.n_img))
        self.canvas.delete(self.img_id)
        for bbox in self.drawn_bboxes:
            self.canvas.delete(bbox.itemId)
        self.entoboxes[self.current].show(self)
        self.update_count()
    def unselect(self):
        for bbox in self.selected:
            bbox.redraw(self)
        self.selected = []

    def rate_g(self):
        for box in self.selected:
            box.status = CONFIRMED
            box.redraw(gui)
        self.unselect()
        self.update_count()
    def rate_b(self):
        for box in self.selected:
            box.status = REJECTED
            box.redraw(gui)
        self.unselect()
        self.update_count()

    def update_thresh(self,val):
        val = int(float(val))
        self.conf_threshold = float(val)/100
        self.thresh_label.config(text= "Confidence threshold: "+ str(val)+"%")
        for bbox in self.entoboxes[self.current].bboxes:
            if bbox.status == DOUBT and bbox.conf > self.conf_threshold:
                bbox.status = SURE
            elif bbox.status == SURE and bbox.conf < self.conf_threshold:
                bbox.status = DOUBT 
        self.change_img(0) #Redraws current entobox 
        self.update_count()
    def update_count(self):
        cnt = 0
        for bbox in self.entoboxes[self.current].bboxes:
            if bbox.status == CONFIRMED or bbox.status == SURE:
                cnt += 1
        self.number_label.config(text= str(cnt)+" speciments detected")

    def on_click(self,e):  #Drawing boxes is in here

        if self.drawing == 0:
            self.select(e)
        elif self.drawing == 1:
            self.draw_coord = [e.x,e.y]
            self.draw_indic = self.canvas.create_oval(self.draw_coord[0],self.draw_coord[1],self.draw_coord[0],self.draw_coord[1],fill=COLORS[SELECTED],outline=COLORS[SELECTED],width=4)
            self.drawing = 2
        elif self.drawing == 2:
            self.canvas.delete(self.draw_indic)
            x1 = min(self.draw_coord[0],e.x)
            x2 = max(self.draw_coord[0],e.x)
            y1 = min(self.draw_coord[1],e.y)
            y2 = max(self.draw_coord[1],e.y)
            if self.drawing_reason == NEW_BBOX:
                new = BBox([x1,y1,x2,y2],1,self.conf_threshold,self.entoboxes[self.current])
                new.status = CONFIRMED
                self.entoboxes[self.current].bboxes.append(new)
                new.draw(self)
                self.update_count()
            elif self.drawing_reason == SELECTING:
                self.unselect()
                for bbox in self.entoboxes[self.current].bboxes:
                    c = bbox.coord
                    if(x1<c[0] and y1<c[1] and x2>c[2] and y2>c[3]):
                        self.selected.append(bbox)
                        self.canvas.itemconfig(bbox.itemId,outline = COLORS[SELECTED])
            self.drawing = 0
            self.drawing_reason = 0

    def select(self,e,cumul = False):
        found = None

        for bbox in self.entoboxes[self.current].bboxes:
            c = bbox.coord 
            if (e.x>c[0] and e.y>c[1] and e.x<c[2] and e.y < c[3] and bbox not in self.selected):
                self.canvas.itemconfig(bbox.itemId,outline = COLORS[SELECTED])
                found = bbox
                break

        if not cumul:
            self.unselect()

        if found != None:
            if cumul:
                self.selected.append(found)
            else:
                self.selected = [found]

    def select_many(self,e):
        self.select(e,True)
    def select_group(self):
        self.start_draw(reason=SELECTING)

    def save(self):
        missing_tag = False
        for bbox in self.entoboxes[self.current].bboxes:
            if bbox.status == DOUBT:
                missing_tag = True
                break
        if missing_tag:
            self.save_label.config(text="Save Failed, Unresolved Tags")
            return
        
        if self.classes == []:
            self.get_classes()
        
        lf = open(os.path.join(self.save_dir,"classes.txt"),"a")
        f = open(os.path.join(self.save_dir,self.entoboxes[self.current].name)+".txt","w")

        for bbox in self.entoboxes[self.current].bboxes:
            if bbox.status != REJECTED:
                if bbox.label not in self.classes:
                    lf.write(bbox.label+"\n")
                    self.classes.append(bbox.label)
                cnum = self.classes.index(bbox.label)
                f.write(str(cnum)+" "+" ".join(str(x) for x in bbox.to_yolo())+ "\n")

        f.close()
        lf.close()

        self.save_label.config(text="Save Successful")

    def get_classes(self):
        name = os.path.join(self.save_dir,"classes.txt")
        if not os.path.isfile(name):
            return
        else:
            lf = open(name,"r")
            self.classes = []
            for line in lf:
                self.classes.append(line)
            lf.close()
            
    def combine(self):
        if len(self.selected)<2:
            return
        coord = self.selected[0].coord.copy()
        for bbox in self.selected:
            if bbox.coord[0] < coord[0]:
                coord[0] = bbox.coord[0]
            if bbox.coord[1] < coord[1]:
                coord[1] = bbox.coord[1]
            if bbox.coord[2] > coord[2]:
                coord[2] = bbox.coord[2]
            if bbox.coord[3] > coord[3]:
                coord[3] = bbox.coord[3]
        
        self.entoboxes[self.current].bboxes = [box for box in self.entoboxes[self.current].bboxes if box not in self.selected]
        for bbox in self.selected:
            self.canvas.delete(bbox.itemId)
        self.selected = []
        new = BBox(coord,1,self.conf_threshold,self.entoboxes[self.current])
        new.status = CONFIRMED
        new.draw(self)
        self.entoboxes[self.current].bboxes.append(new)    
        self.unselect()
        self.update_count()
    def add_label(self):
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
            for bbox in self.selected:
                bbox.label = e.get()
            label_window.destroy()

        ttk.Button(tfrm,text="Ok",command=conf_label).grid(row=2,column=0)

    def start_draw(self,reason = NEW_BBOX):
        self.drawing_reason = reason
        self.drawing = 1
    
    def get_dim(self,dim):
        x = dim[0]
        y = dim[1]
        scale = min(self.x_max/x,self.y_max/y)
        return(int(x*scale),int(y*scale))



if __name__ == "__main__":


    gui = GUI()
    img_path = "MiniSample"




    

    gui.root.mainloop()