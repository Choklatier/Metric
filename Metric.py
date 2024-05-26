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

# Metric class stores the metric as [[a,b],[c,d]]
# a,b,c,d can be functions of x and y.
class Metric:
    
    def __init__(self,a,b,c,d):
        self.a,self.b,self.c,self.d = a,b,c,d
        self.params = [a,b,c,d]

    def __call__(self,v1,v2,x = 0,y = 0):
        # stores the params temporarily
        # compute functions if params depend on x,y
        p = [] 
        for param in self.params:
            if callable(param):
                p.append(param(x,y))
            else :
                p.append(param)
        M = np.array([[p[0],p[1]],
                      [p[2],p[3]]])
        return v2 @ (M @ v1)
    
    def plot_distance_contour(self,
                              xmin = -1,xmax = 1,
                              ymin = -1,ymax = 1,nb_points = 100,
                              axis = plt):
        x = np.linspace(xmin,xmax,nb_points)
        y = np.linspace(ymin,ymax,nb_points)
        #X,Y = np.meshgrid(x,y)
        Z = np.zeros((nb_points,nb_points))
        for i in range(nb_points):
            for j in range(nb_points):
                v = np.array([x[i],y[j]])
                Z[j,i] = self(v,v,x[i],y[j])
  
        axis.contourf(x,y,np.sqrt(Z))
        

# Class that holds points coordinates for each line
class Line:

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
        return Line(new_x,new_y)
    
    def rotate(self,angle):
        
        new_x = self.x * np.cos(angle * np.pi/180) + self.y * np.sin(angle * np.pi/180)
        new_y = self.x * np.sin(angle * np.pi/180) - self.y * np.cos(angle * np.pi/180)
        self.x = new_x
        self.y = new_y

# Class that generate and manages line objects
class Grid:

    def __init__(self,
                 xmin = -1,xmax = 1,
                 ymin = -1,ymax = 1,
                 nb_lines = 6,nb_points = 100,
                 color = None):

        self.lines = []
        # Generate horizontal lines
        for constant_y in np.linspace(ymin,ymax,nb_lines):
            self.lines.append(Line(x = np.linspace(xmin,xmax,nb_points),
                                   y = constant_y * np.ones(nb_points)))
        # Generate vertical lines
        for constant_x in np.linspace(ymin,ymax,nb_lines):
            self.lines.append(Line(x = constant_x * np.ones(nb_points),
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
            
if __name__ == "__main__":
    
    fig,axes = plt.subplots(2,2)

    g = Grid(nb_lines = 5)
    g.plot(axis = axes[0,0])

    g2 = Grid(nb_lines = 5,color = "black")
    g2.rotate(45)
    g2.plot(axis = axes[0,0])
    
    def f1(x,y):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)
        # DND
        #return 1 + np.abs(np.cos(2 * phi))
        rs = 0.1
        # SC
        #return -1/(1 - (rs/(x-0.5)))
        # SC lemaitre
        #r = (1.5 * (x - y))**(2/3) * rs**(1/3)
        #return -rs/r
        #
        
        return -1

    def f2(x,y):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)
        #return 1 + np.abs(np.cos(2 * phi))
        rs = 0.1
        #return 1 - (rs/(x-0.5))
        #r = (1.5 * (x - y))**(2/3) * rs**(1/3)
        vs = 1.2
        rs = np.sqrt((x - y)**2)
        sig = 0.1
        R = 0.5
        f = np.tanh(sig * (rs + R)) - np.tanh(sig*(rs - R))/(2*np.tanh(sig * R))
        return 1 - vs**2 * f**2

    def f3(x,y):
        vs = 1.2
        rs = np.sqrt((x - y)**2)
        sig = 0.1
        R = 0.5
        f = np.tanh(sig * (rs + R)) - np.tanh(sig*(rs - R))/(2*np.tanh(sig * R))
        return vs * f

    I = Metric(1,0,0,1)
    #m = Metric(-1,0,0,1)
    m = Metric(f1,f3,f3,f2)

    g.transform(m,axis = axes[0,1])
    g2.transform(m,axis = axes[0,1])

    I.plot_distance_contour(axis = axes[1,0])
    m.plot_distance_contour(axis = axes[1,1])
    
    #plt.savefig("test.png")
    plt.show()
