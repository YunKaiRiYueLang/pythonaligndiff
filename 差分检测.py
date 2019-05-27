
from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter.ttk import Treeview

import tkinter.filedialog
root =Tk()
root.title("imageprocess")
root.resizable(False,False)#固定大小







  
  
templateimagepath=""
templateimage=0
srcimage=""
srcpath=""
loadsrc=None
tkloadsrc=None
tktemsrc=None
tkpreview=None
diffimg=None  
tkalign=None
canvas1=Canvas(root,height=500,width=700)
canvas2=Canvas(root,height=500,width=700)
canvas3=Canvas(root,height=500,width=800)

def openFileEvent():
   
    global templateimagepath
    templateimagepath = tkinter.filedialog.askopenfilename(initialdir='C:\Python27',
        initialfile='README',title='打开新文件')
    if len(templateimagepath) == 0:
        print("未选择图片")
        return
    # 显示新路径内容
    print(templateimagepath)
#从摄像头抓取一张图,并显示在界面中
def preview_hit():
    cap=cv2.VideoCapture(0)#打开默认相机,有一个额外的相机,打开额外相机一把数字为1
    if(cap.isOpened()==False):
        print("没有正确打开相机")
        return
    while(cap.isOpened()):
        s,img=cap.read()# 读取的图像为img
        global srcimage
        srcimage=image
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        
        global tkpreview
        tkpreview=ImageTk.PhotoImage(image=image)
        canvas1.create_image(0,0,anchor='nw',image=tkpreview)
        break
        ##print(len(img))
        #cv2.imshow("test",img)
        #k=cv2.waitKey(0)
        #global srcimage
        #if k==ord('c'):
            #cv2.imwrite("D:/001TEST.jpg",img)
        #if k==ord('q'):
            #break
    cap.release()
    #cv2.destroyAllWindows()
    
def loadtemplate_hit():
    openFileEvent()
    if len(templateimagepath)==0:
        return 
    global templateimage
    templateimage=cv2.imread(templateimagepath);
    if templateimage is None:
        print("kong tu xaing ")
        return
    tem=Image.open(templateimagepath)
    global tktemsrc
    tktemsrc=ImageTk.PhotoImage(image=tem)
    canvas2.create_image(0,0,anchor='nw',image=tktemsrc)
    #cv2.imshow("tem",templateimage)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
def loadsrc_hit():
    
    global srcpath;
    srcpath = tkinter.filedialog.askopenfilename(initialdir='C:\Python27',
        initialfile='选择一副图',title='打开新文件')
    global srcimage
    if len(srcpath)==0:
        print("wei xuan ze tu pian ")
        return
    srcimage=cv2.imread(srcpath)
    if srcimage is None:
        return 
    loadsrc=Image.open(srcpath)
    global tkloadsrc
    tkloadsrc=ImageTk.PhotoImage(image=loadsrc)
    global canvas1
    canvas1.create_image(0,0,anchor='nw',image=tkloadsrc)
    #cv2.imshow("src",srcimage)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    

def alignImages():
    global srcimage
    im1=srcimage
    im2=templateimage
  # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(500)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    srcimage = cv2.warpPerspective(im1, h, (width, height))
    global tkalign
    alignimg=Image.fromarray(cv2.cvtColor(srcimage,cv2.COLOR_BGR2RGB))
    tkalign=ImageTk.PhotoImage(image=alignimg)
    canvas3.create_image(0,0,anchor='nw',image=tkalign)
    #用额外窗口显示配准结果
   # cv2.imshow("aligned",srcimage)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
def pic_sub(dest,s1,s2):
    for x in range(dest.shape[0]):
        for y in range(dest.shape[1]):
            if(s2[x,y] > s1[x,y]):
                dest[x,y] = s2[x,y] - s1[x,y]
            else:
                dest[x,y] = s1[x,y] - s2[x,y]

            if(dest[x,y] < 40):
                dest[x,y] = 0
            else:
                dest[x,y] = 255

    
def diff_hit():
    global srcimage
    srcimagegray= cv2.cvtColor(srcimage,cv2.COLOR_BGR2GRAY)
    templateimagegray=cv2.cvtColor(templateimage,cv2.COLOR_BGR2GRAY)
 #   cv2.imshow("",srcimagegray)
 #   cv2.imshow("",templateimagegray)
    D1=srcimagegray
    pic_sub(D1,srcimagegray,templateimagegray)
    #cv2.imshow("",D1)
    #cv2.waitKey(0)
    #contours,hierarchy =cv2.findContours(D1,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)#这里应该cv版本导致的,在其他地方改的,和我的版本不同
    image, contours, hierarchy =cv2.findContours(D1,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
   
    
    cv2.drawContours(srcimage,contours,-1,(0,0,255),3)
    
    
    global diffimg
    dffim = Image.fromarray(cv2.cvtColor(srcimage,cv2.COLOR_BGR2RGB))
    dffim.show()
    diffimg=ImageTk.PhotoImage(dffim)
    #canvas3.create_image(0,0,anchor='nw',image=diffimg)
    ###
    top = Toplevel()
    top.geometry('1200x501')
    canvas=Canvas(top,bg='red',height=500,width=700)
    canvas.place(x=400,y=0)  
    canvas.create_image(0,0,anchor='nw',image=diffimg)
    #使用Treeview组件实现表格功能
    frame = Frame(top)
    frame.place(x=0, y=10, width=400, height=200)
    #滚动条
    scrollBar = tkinter.Scrollbar(frame)
    scrollBar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
    #Treeview组件，6列，显示表头，带垂直滚动条
    tree = Treeview(frame,
                              columns=('c1', 'c2', 'c3'),                              
                              show="headings",                              
                              yscrollcommand=scrollBar.set)
    #设置每列宽度和对齐方式    
    tree.column('c1', width=100, anchor='center')
    tree.column('c2', width=100, anchor='center')
    tree.column('c3', width=100, anchor='center')
    #设置每列表头标题文本
    tree.heading('c1', text='序号')
    tree.heading('c2', text='位置')
    tree.heading('c3', text='面积')
    tree.place(x=0,y=0)
    #tree.pack(side=tkinter.LEFT, fill=tkinter.Y)
    #Treeview组件与垂直滚动条结合
    scrollBar.config(command=tree.yview)
    #定义并绑定Treeview组件的鼠标单击事件
    def treeviewClick(event):
        pass
    tree.bind('<Button-1>', treeviewClick)
    
    aaa=len(contours)
    print(aaa)
    
    if aaa==0:
        return
    #插入演示数据
    for i in range(aaa):
        cnt = contours[i]
        M = cv2.moments(cnt)

        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        area = cv2.contourArea(cnt)
        print(area)
        tcx='%d' %cx
        tcy='%d' %cy
        #tarea='%d' %area
        tree.insert('', i, values=(i,tcx+','+tcy,area))
    
    


def signinbtn():
    print('test')
    ybutton.destroy()
    canvas.delete('all')
    canvas.destroy()
    root.geometry('1500x1100')
    
    
    canvas1.place(x=800,y=0)
    
    
    canvas2.place(x=800,y=501)
    
    
    canvas3.place(x=0,y=501)
    
    
    loadtemplate=Button(root,text="建立模板",command=loadtemplate_hit,height=1,width=15)
    loadtemplate.pack()
    loadtemplate.place(x=100,y=50)


    loadsrcimage=Button(root,text="加载图片",command=loadsrc_hit,height=1,width=15)
    loadsrcimage.pack()
    loadsrcimage.place(x=100,y=100)



    recsrcimage=Button(root,text="Registrationin",command=alignImages,height=1,width=15)
    recsrcimage.pack()
    recsrcimage.place(x=100,y=150)

    previes=Button(root,text="获取图片",command=preview_hit,height=1,width=15)
    previes.pack()
    previes.place(x=100,y=1)

    difference=Button(root,text="difference",command=diff_hit,height=1,width=15)
    difference.pack()
    difference.place(x=100,y=200)
    
root.geometry('1000x600')
canvas=Canvas(root,height=600,width=1000)
canvas.place(x=0,y=0)
lena=Image.open('lena.jpg')
tklena=ImageTk.PhotoImage(image=lena)
canvas.create_image(0,0,anchor='nw',image=tklena)
ybutton=Button(root,text='登陆',height=1,width=15,command=signinbtn)
ybutton.place(x=850,y=500)
  



root.mainloop()
