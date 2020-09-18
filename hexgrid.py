from tkinter import *
import numpy as np
from PIL import ImageGrab
# Values for grid label
label_dict = {
	"High Income" : "Red", 		        # High income countries
	"Middle High Income" : "Orange",	# Medium High income countries
	"Middle Low Income" : "Yellow",	    # Medium Low income countries
	"Low Income" : "Green"		        # Low income countries
}

class HexaCanvas(Canvas):
    def __init__(self, master, *args, **kwargs):
        Canvas.__init__(self, master, *args, **kwargs)
        self.hexaSize = 20
    def setHexaSize(self, number):
        self.hexaSize = number

    def create_hexagon(self, x, y,fill="",text="",text_with_color = {}):

        size = self.hexaSize
        lineColor = "white"
        font_size = 10

        
        dx = (size**2 - (size/2)**2)**0.5

        point1 = (x+dx, y+size/2)
        point2 = (x+dx, y-size/2)
        point3 = (x   , y-size  )
        point4 = (x-dx, y-size/2)
        point5 = (x-dx, y+size/2)
        point6 = (x   , y+size  )

        # self.create_text(y,x,text=text)
        self.create_line(point1, point2, fill=lineColor, width=2)
        self.create_line(point2, point3, fill=lineColor, width=2)
        self.create_line(point3, point4, fill=lineColor, width=2)
        self.create_line(point4, point5, fill=lineColor, width=2)
        self.create_line(point5, point6, fill=lineColor, width=2)
        self.create_line(point6, point1, fill=lineColor, width=2)
        
        if fill != None:
            self.create_polygon(point1, point2, point3, point4, point5, point6, fill=fill)

        if text_with_color == None:
            self.create_text(x,y,font=("Arial", font_size),text=text,fill=textcolor)

        else:

            text_size = len(text_with_color)
            
            y_decrease = 4 if text_size > 3 else text_size

            offset = 0
            for item in list(text_with_color.items())[:3]:
                self.create_text(x,y-y_decrease*4+offset,font=("Arial", font_size),text=item[0],fill=item[1])
                offset += 10
            if(text_size > 3):
                self.create_text(x,y-y_decrease*4+offset,font=("Arial", font_size),text='+{}'.format(text_size-3),fill="white")
 
        


class HexagonalGrid(HexaCanvas):
    def __init__(self, master, scale, grid_width, grid_height, *args, **kwargs):

        dx     = (scale**2 - (scale/2.0)**2)**0.5
        width  = 2 * dx * grid_width + dx
        height = 1.5 * scale * grid_height + 0.5 * scale

        HexaCanvas.__init__(self, master, background='white', width=width, height=height, *args, **kwargs)
        self.setHexaSize(scale)

    def setCell(self, xCell, yCell, *args, **kwargs ):
        #compute pixel coordinate of the center of the cell:
        size = self.hexaSize
        dx = (size**2 - (size/2)**2)**0.5

        pix_x = dx + 2*dx*xCell
        if yCell%2 ==1 :
            pix_x += dx

        pix_y = size + yCell*1.5*size

        self.create_hexagon(pix_x, pix_y, *args, **kwargs)



class SomGrid():
    def __init__(self, grid_width, grid_height,dict,distance,colors_dict):
        self.width = grid_width
        self.height = grid_height
        self.neuronsDict = dict
        self.distance_map = distance
        self.colors_dict = colors_dict
        self.generateGrid()

    def generateGrid(self):
        self.createGrid()
        self.generateUMatrix()
        self.addHexagons()

    def createGrid(self):
        self.tk = Tk()
        self.tk.title("Self Organizing Maps")
        self.setLabel()
        self.grid = HexagonalGrid(self.tk, scale = 32, grid_width=self.width, grid_height=self.height)
        self.grid.grid(row=1, column=0, padx=5, pady=5)
        quit = Button(self.tk, text = "Quit", command = lambda :self.correct_quit())
        quit.grid(row=2, column=0)
    def setLabel(self):
        text = ""
        for i,(key,value) in enumerate(label_dict.items()):
            text += '     {}: {}     |'.format(value,key)
        label = Label(self.tk, text=text, fg="black", font="Times")
        label.grid(row=0,column=0)

    def generateUMatrix(self):
        for i in range(0,self.width):
            for j in range(0,self.height):
                self.grid.setCell(i,j,self.distance_map[i][j])

    def addHexagons(self, color=""):
        # Add the winners 
        for key,value in self.neuronsDict.items():
            text = {code: self.colors_dict[code] for code in value}
            self.grid.setCell(key[0],key[1],text_with_color=text)

    def display(self):
        self.tk.mainloop()

    def correct_quit(self):
        self.tk.destroy()
        self.tk.quit()

