from tkinter import *
from PIL import ImageTk,Image
import numpy as np
import torch
import sys

#----------------------------------------------------------------------

class MainWindow():

	#----------------

	def __init__(self,main):

	#canvasforimage
	#main.bind("<Button1>",on)
		self.path = "D:\\Datasets\\Head_hunt\\"
		self.x0=None
		self.y0=None
		self.cords=[]
		self.canvas=Canvas(main,width=500,height=500)
		self.canvas.grid(row=0,column=0)
		self.dataset_size = 1748
		#images
		self.my_images=[]
		for i in range(1, self.dataset_size):
			image = Image.open(self.path + "head ("+str(i)+").png")
			image = image.resize((500, 500), Image.ANTIALIAS)
			self.my_images.append(ImageTk.PhotoImage(image))
	
		self.my_image_number=0

		#setfirstimageoncanvas
		self.image_on_canvas=self.canvas.create_image(0, 0, anchor=NW,image=self.my_images[self.my_image_number])

		#buttontochangeimage
	
	

	#----------------
	def onLclick(self, event):

		self.x0 = event.x
		self.y0 = event.y
		self.cords.append([self.x0//5, self.y0//5, 1])
		print(self.x0//5, self.y0//5, 1)
		#nextimage
		self.my_image_number+=1
		#returntofirstimage
		if self.my_image_number==len(self.my_images):
			print(self.cords)
			self.cords = np.array(self.cords)
			np.savetxt('data.csv', self.cords, delimiter=',')
			sys.exit()
		#savecords
		#changeimage2222
		self.canvas.itemconfig(self.image_on_canvas, image = self.my_images[self.my_image_number])

	def onRclick(self, event):

		self.cords.append([-1, -1, -1])
		print(-1, -1, -1)
		#nextimage
		self.my_image_number+=1
		#returntofirstimage
		if self.my_image_number==len(self.my_images):
			print(self.cords)
			self.cords = np.array(self.cords)
			np.savetxt('data.csv', self.cords, delimiter=',')
			sys.exit()
		#savecords
		#changeimage
		self.canvas.itemconfig(self.image_on_canvas, image = self.my_images[self.my_image_number])

root=Tk()
C = MainWindow(root)
root.bind("<Button-1>", C.onLclick)
root.bind("<Button-3>", C.onRclick)
root.mainloop()