import numpy as np
import matplotlib.pyplot as plt

def xy_to_polar(x,y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)
    return r,phi

def polar_to_xy(r,phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x,y

# Class that holds points coordinates for each line
class Curve:

    def __init__(self,x,y):
        if len(x) != len(y):
                print("Error, x and y different lengths")
                return
        
        self.x = x
        self.y = y
        self.nb_points = len(x)

    # Takes in a Metric Object
    def transform(self,metric):
        # TODO : figure out how I can add a dimension to the metric
        # to get rid of for loop
        new_r = []
        for i in range(self.nb_points):
            v = np.array([self.x[i],self.y[i]])
            r_sq = metric(v,v,v[0],v[1])
            new_r.append(np.sqrt(r_sq))
        
        new_r = np.array(new_r)
        # get current stuff
        _,phi = xy_to_polar(self.x,self.y)

        new_x,new_y = polar_to_xy(new_r,phi)
        return Curve(new_x,new_y)
    
    def rotate(self,angle):
        
        new_x = self.x * np.cos(angle * np.pi/180) + self.y * np.sin(angle * np.pi/180)
        new_y = self.x * np.sin(angle * np.pi/180) - self.y * np.cos(angle * np.pi/180)
        self.x = new_x
        self.y = new_y

# Class that generate and manages lines stored in curve objects
class Grid:

    def __init__(self,
                 xmin = -1,xmax = 1,
                 ymin = -1,ymax = 1,
                 nb_lines = 6,nb_points = 100,
                 color = None):

        self.lines = []
        # Generate horizontal lines
        for constant_y in np.linspace(ymin,ymax,nb_lines):
            self.lines.append(Curve(x = np.linspace(xmin,xmax,nb_points),
                                   y = constant_y * np.ones(nb_points)))
        # Generate vertical lines
        for constant_x in np.linspace(ymin,ymax,nb_lines):
            self.lines.append(Curve(x = constant_x * np.ones(nb_points),
                                   y = np.linspace(ymin,ymax,nb_points)))

        self.color = color

    def plot(self,axis = plt):

        for line in self.lines:
            if self.color == None:
                axis.plot(line.x,line.y)
            else:
                axis.plot(line.x,line.y,color = self.color)

    def set_lines(self,lines):
        self.lines = lines
    
    # Uses a metric object to transform lines
    def transform(self,metric,axis = plt):
        
        for line in self.lines:
            new_line = line.transform(metric)
            if self.color == None:
                axis.plot(new_line.x,new_line.y)
            else:
                axis.plot(new_line.x,new_line.y,color = self.color)
    
    def rotate(self,angle):
        
        for line in self.lines:
            line.rotate(angle)

class CircularGrid:
    
    def __init__(self,
                 r_min = 0.1,r_max = 1,
                 phi_min = 0,phi_max = 2 * np.pi,
                 nb_curves = 6,nb_points = 100,
                 color = "green"):
    
        self.circles = []
        for r in np.linspace(r_min,r_max,nb_curves):
            phi = np.linspace(phi_min,phi_max,nb_points) 
            
            self.circles.append(Curve(*polar_to_xy(r * np.ones(len(phi)),phi)))

        self.color = color
    
    def plot(self,axis = plt):

        for circle in self.circles:
            if self.color == None:
                axis.plot(circle.x,circle.y)
            else:
                axis.plot(circle.x,circle.y,color = self.color) 

    # Uses a metric object to transform lines
    def transform(self,metric,axis = plt):
        
        for circle in self.circles:
            new_circle = circle.transform(metric)
            if self.color == None:
                axis.plot(new_circle.x,new_circle.y)
            else:
                axis.plot(new_circle.x,new_circle.y,color = self.color)
