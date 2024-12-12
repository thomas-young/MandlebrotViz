import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, 
                             QHBoxLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QRect, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
import numpy as np
from matplotlib import cm
from numba import njit, prange
from matplotlib.colors import LinearSegmentedColormap
import time

@njit(parallel=True)
def mandelbrot_iteration(z, c, div_time, iteration, max_iter):
    height, width = z.shape
    for i2 in prange(height):
        for j2 in range(width):
            if div_time[i2,j2]==0:
                z_val=z[i2,j2]
                c_val=c[i2,j2]
                z_val=z_val*z_val+c_val
                if (z_val.real*z_val.real+z_val.imag*z_val.imag)>4.0:
                    div_time[i2,j2]=iteration
                    z_val=2+0j
                z[i2,j2]=z_val
    return z, div_time

@njit(parallel=True)
def julia_iteration(z, c_constant, div_time, iteration, max_iter):
    height,width=z.shape
    for i2 in prange(height):
        for j2 in range(width):
            if div_time[i2,j2]==0:
                z_val=z[i2,j2]
                z_val=z_val*z_val+c_constant
                if (z_val.real*z_val.real+z_val.imag*z_val.imag)>4.0:
                    div_time[i2,j2]=iteration
                    z_val=2+0j
                z[i2,j2]=z_val
    return z, div_time

@njit(parallel=True)
def burning_ship_iteration(z,c,div_time,iteration,max_iter):
    height,width=z.shape
    for i2 in prange(height):
        for j2 in range(width):
            if div_time[i2,j2]==0:
                z_val=z[i2,j2]
                c_val=c[i2,j2]
                z_val=complex(abs(z_val.real),abs(z_val.imag))
                z_val=z_val*z_val+c_val
                if (z_val.real*z_val.real+z_val.imag*z_val.imag)>4.0:
                    div_time[i2,j2]=iteration
                    z_val=2+0j
                z[i2,j2]=z_val
    return z, div_time

@njit(parallel=True)
def newton_iteration(z,div_time,iteration,max_iter):
    # Newton fractal: f(z)=z³-1
    # roots: 1, -0.5+0.86602540378j, -0.5-0.86602540378j
    roots = np.array([1+0j, -0.5+0.86602540378j, -0.5-0.86602540378j],dtype=np.complex128)
    tol=1e-7
    height,width=z.shape
    for i2 in prange(height):
        for j2 in range(width):
            if div_time[i2,j2]==0:
                z_val=z[i2,j2]
                f=z_val*z_val*z_val-1
                fprime=3*z_val*z_val
                if fprime==0:
                    div_time[i2,j2]=iteration
                    z_val=2+0j
                else:
                    z_val=z_val - f/fprime
                    for r_i,root in enumerate(roots):
                        if abs(z_val-root)<tol:
                            div_time[i2,j2]=iteration+(r_i+1)*max_iter
                            break
                z[i2,j2]=z_val
    return z, div_time

def gaussian_blur(div_time_cpu):
    # A small 3x3 Gaussian blur kernel
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float64)
    kernel = kernel / kernel.sum()

    # Apply convolution ignoring edges for simplicity
    # We won't be super strict since we only want slight smoothing
    h, w = div_time_cpu.shape
    out = div_time_cpu.copy()
    for i in range(1, h-1):
        for j in range(1, w-1):
            region = div_time_cpu[i-1:i+2, j-1:j+2]
            val = (region * kernel).sum()
            out[i,j] = val
    return out

class FractalWorker(QThread):
    finished = pyqtSignal(QImage)
    partialUpdate = pyqtSignal(QImage)

    # ADDED: Accept c_constant as a parameter from outside.
    def __init__(self, x_min, x_max, y_min, y_max, width, height, max_iter, fractal_type, c_constant=None):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.fractal_type = fractal_type
        self.move_on = True

        # If no c_constant provided, use default
        if c_constant is not None:
            self.c_constant = c_constant
        else:
            if fractal_type==0: # mandelbrot
                self.c_constant=0+0j
            elif fractal_type==1:
                self.c_constant = -0.8+0.156j
            elif fractal_type==2:
                self.c_constant = 0.285+0.01j
            elif fractal_type==3: # burning ship
                self.c_constant=0+0j
            elif fractal_type==4: # newton
                self.c_constant=0+0j
            elif fractal_type==5:
                self.c_constant=-0.4+0.6j

    def _create_image(self, div_time_cpu, blur=False):
        if blur:
            div_time_cpu = gaussian_blur(div_time_cpu)
        dmin=div_time_cpu.min()
        dmax=div_time_cpu.max()
        if dmax>dmin:
            norm=(div_time_cpu-dmin)/(dmax-dmin)
        else:
            norm=np.zeros_like(div_time_cpu,dtype=np.float64)
 

        cdict = [
            (0.0,    (  0/255.0,   7/255.0, 100/255.0)),
            (0.16,   ( 32/255.0, 107/255.0, 203/255.0)),
            (0.42,   (237/255.0, 255/255.0, 255/255.0)),
            (0.6425, (255/255.0, 170/255.0,   0/255.0)),
            (1.0,    (  0/255.0,   2/255.0,   0/255.0))
        ]

        # Create the colormap from the listed colors
        cmap = LinearSegmentedColormap.from_list("custom_mandelbrot", cdict)

        rgba=cmap(norm)[:,:,:3]
        rgba=(rgba*255).astype('uint8')
        image=QImage(rgba.data,self.width,self.height,3*self.width,QImage.Format_RGB888)
        return image

    def run(self):
        print("Starting fractal calculation of type:", self.fractal_type)
        x = np.linspace(self.x_min, self.x_max, self.width)
        y = np.linspace(self.y_max, self.y_min, self.height)
        if self.fractal_type in (0,3): # mandelbrot or burning ship
            c = x[np.newaxis,:]+1j*y[:,np.newaxis]
            z = np.zeros((self.height,self.width),dtype=np.complex128)
            div_time = np.zeros((self.height,self.width),dtype=np.int32)
        elif self.fractal_type in (1,2,5): # julia
            xx,yy=np.meshgrid(x,y)
            z=xx+1j*yy
            c=np.zeros((self.height,self.width),dtype=np.complex128)
            div_time=np.zeros((self.height,self.width),dtype=np.int32)
        elif self.fractal_type==4: # newton
            xx,yy=np.meshgrid(x,y)
            z=xx+1j*yy
            c=np.zeros((self.height,self.width),dtype=np.complex128)
            div_time=np.zeros((self.height,self.width),dtype=np.int32)

        update_interval=1
        
        for i in range(self.max_iter):
            if not self.move_on:
                print("Calculation stopped early.")
                return
            iteration=i+1
            if self.fractal_type==0: # mandelbrot
                z,div_time=mandelbrot_iteration(z,c,div_time,iteration,self.max_iter)
            elif self.fractal_type in (1,2,5): # julia sets
                z,div_time=julia_iteration(z,self.c_constant,div_time,iteration,self.max_iter)
            elif self.fractal_type==3: # burning ship
                z,div_time=burning_ship_iteration(z,c,div_time,iteration,self.max_iter)
            elif self.fractal_type==4: # newton
                z,div_time=newton_iteration(z,div_time,iteration,self.max_iter)


            if iteration%update_interval==0 and self.move_on and iteration > 10:
                # Emit partial update (no blur on partial to keep it responsive)
                temp=div_time.copy()
                temp[temp==0]=iteration
                partial_img=self._create_image(temp, blur=False)
                self.partialUpdate.emit(partial_img)

        div_time[div_time==0]=self.max_iter
        final_image=self._create_image(div_time, blur=True)  # apply blur at the end
        if self.move_on:
            self.finished.emit(final_image)


class ClickableLabel(QLabel):
    selectionComplete=pyqtSignal(QPoint,QPoint)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.start_pos=None
        self.current_pos=None
        self.drawing=False
        self.setScaledContents(False)
        self.setMouseTracking(True)

    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton:
            self.start_pos=event.pos()
            self.current_pos=event.pos()
            self.drawing=True
            self.update()

    def mouseMoveEvent(self,event):
        if self.drawing:
            self.current_pos=event.pos()
            self.update()

    def mouseReleaseEvent(self,event):
        if event.button()==Qt.LeftButton and self.drawing:
            self.drawing=False
            end_pos=event.pos()
            if (abs(end_pos.x()-self.start_pos.x())>10 and abs(end_pos.y()-self.start_pos.y())>10):
                self.selectionComplete.emit(self.start_pos,end_pos)
            self.start_pos=None
            self.current_pos=None
            self.update()

    def paintEvent(self,event):
        super().paintEvent(event)
        if self.start_pos and self.current_pos and self.drawing:
            painter=QPainter(self)
            pen=QPen(QColor(255,0,0,200))
            pen.setWidth(2)
            painter.setPen(pen)
            brush=QColor(255,0,0,50)
            painter.setBrush(brush)
            rect=QRect(self.start_pos,self.current_pos)
            painter.drawRect(rect)


class FractalWindow(QWidget):
    def __init__(self):
        super().__init__()
        screen=QApplication.primaryScreen().availableGeometry()
        screen_width,screen_height=screen.width(),screen.height()

        self.desired_width=1600
        self.desired_height=1200
        self.width=min(self.desired_width,int(screen_width*0.9))
        self.height=min(self.desired_height,int(screen_height*0.9))

        self.base_iter=5000
        self.max_iter=self.base_iter

        self.initial_x_min=-1.5
        self.initial_x_max=1.5
        self.initial_y_min=-1.0
        self.initial_y_max=1.0

        self.x_min=self.initial_x_min
        self.x_max=self.initial_x_max
        self.y_min=self.initial_y_min
        self.y_max=self.initial_y_max

        self.zoom_level=0
        self.pixel_threshold=1e-14

        self.fractal_type=0 # start with mandelbrot

        # ADDED: Store default Julia parameters and track current one
        self.julia_params = {
            1: complex(-0.8, 0.156),
            2: complex(0.285, 0.01),
            5: complex(-0.4, 0.6)
        }

        self.setWindowTitle("Fractal Viewer")
        main_layout=QVBoxLayout(self)

        # Add buttons for fractals
        fractal_bar=QHBoxLayout()
        btn_mandel=QPushButton("Mandelbrot")
        btn_mandel.clicked.connect(lambda:self.switch_fractal(0))
        fractal_bar.addWidget(btn_mandel)

        btn_j1=QPushButton("Julia #1")
        btn_j1.clicked.connect(lambda:self.switch_fractal(1))
        fractal_bar.addWidget(btn_j1)

        btn_j2=QPushButton("Julia #2")
        btn_j2.clicked.connect(lambda:self.switch_fractal(2))
        fractal_bar.addWidget(btn_j2)

        btn_burn=QPushButton("Burning Ship")
        btn_burn.clicked.connect(lambda:self.switch_fractal(3))
        fractal_bar.addWidget(btn_burn)

        btn_newton=QPushButton("Newton")
        btn_newton.clicked.connect(lambda:self.switch_fractal(4))
        fractal_bar.addWidget(btn_newton)

        btn_j3=QPushButton("Julia #3")
        btn_j3.clicked.connect(lambda:self.switch_fractal(5))
        fractal_bar.addWidget(btn_j3)

        main_layout.addLayout(fractal_bar)

        self.label=ClickableLabel("Draw a rectangle to zoom")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(self.width,self.height)
        main_layout.addWidget(self.label)

        self.button=QPushButton("Generate Fractal")
        self.button.clicked.connect(self.start_calculation)
        main_layout.addWidget(self.button)

        self.label.selectionComplete.connect(self.handle_rectangle_selection)

        self.worker=None
        QTimer.singleShot(0,self.start_calculation)
        self.setLayout(main_layout)

    def keyPressEvent(self,event):
        if event.key()==Qt.Key_Space:
            self.abort_current_worker_if_running()
            self.x_min=self.initial_x_min
            self.x_max=self.initial_x_max
            self.y_min=self.initial_y_min
            self.y_max=self.initial_y_max
            self.zoom_level=0
            self.max_iter=self.base_iter
            self.start_calculation()

        elif event.key() == Qt.Key_Left:
            if self.fractal_type in (1,2,5):
                self.abort_current_worker_if_running()  # Ensure no current calc
                c = self.julia_params[self.fractal_type]
                self.julia_params[self.fractal_type] = complex(c.real - 0.01, c.imag)
                self.start_calculation()
                
        elif event.key() == Qt.Key_Right:
            if self.fractal_type in (1,2,5):
                self.abort_current_worker_if_running()  # Ensure no current calc
                c = self.julia_params[self.fractal_type]
                self.julia_params[self.fractal_type] = complex(c.real + 0.01, c.imag)
                self.start_calculation()

    def switch_fractal(self,ftype):
        self.abort_current_worker_if_running()
        self.fractal_type=ftype
        # reset to initial
        self.x_min=self.initial_x_min
        self.x_max=self.initial_x_max
        self.y_min=self.initial_y_min
        self.y_max=self.initial_y_max
        self.zoom_level=0
        self.max_iter=self.base_iter
        self.start_calculation()

    def start_calculation(self):
        if self.worker and self.worker.isRunning():
            return
            
        print("[Main] Start calculation fractal type:", self.fractal_type)
        self.button.setEnabled(False)
        self.label.setText("Calculating...")
        self.run_worker()

    def run_worker(self):
        if self.worker and self.worker.isRunning():
            return
        
        # ADDED: Pass the c_constant if this is a Julia set fractal
        c_constant = None
        if self.fractal_type in (1,2,5):
            c_constant = self.julia_params[self.fractal_type]

        self.worker=FractalWorker(
            x_min=self.x_min,x_max=self.x_max,
            y_min=self.y_min,y_max=self.y_max,
            width=self.width,height=self.height,
            max_iter=self.max_iter,
            fractal_type=self.fractal_type,
            c_constant=c_constant  # pass updated c_constant for Julia sets
        )
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.partialUpdate.connect(self.update_image_partial)
        self.worker.start()

    def on_worker_finished(self,image):
        print("[Main] Worker finished with final image.")
        pix=QPixmap.fromImage(image)
        self.label.setPixmap(pix)
        self.label.setText("")
        self.button.setEnabled(True)
        print("[Main] UI updated with final image.")

    def update_image_partial(self,image):
        # Show partial results (no blur)
        pix=QPixmap.fromImage(image)
        self.label.setPixmap(pix)
        self.label.setText("")

    def abort_current_worker_if_running(self):
        if self.worker and self.worker.isRunning():
            print("[Main] Aborting current calculation...")
            self.worker.move_on=False
            self.worker.wait()
            print("[Main] Current calculation aborted.")

    def handle_rectangle_selection(self, start_pos, end_pos):
        self.abort_current_worker_if_running()

        print(f"[Main] Rectangle selection: start={start_pos}, end={end_pos}")
        x1, x2 = sorted([start_pos.x(), end_pos.x()])
        y1, y2 = sorted([start_pos.y(), end_pos.y()])

        cur_width = self.x_max - self.x_min
        cur_height = self.y_max - self.y_min

        # Compute the raw new coords from the selection
        new_x_min = self.x_min + (x1 / self.width) * cur_width
        new_x_max = self.x_min + (x2 / self.width) * cur_width
        new_y_max = self.y_max - (y1 / self.height) * cur_height
        new_y_min = self.y_max - (y2 / self.height) * cur_height

        # Enforce consistent aspect ratio:
        desired_ratio = self.width / self.height
        current_ratio = (new_x_max - new_x_min) / (new_y_max - new_y_min)

        center_x = (new_x_min + new_x_max) / 2
        center_y = (new_y_min + new_y_max) / 2

        if current_ratio > desired_ratio:
            desired_height = (new_x_max - new_x_min) / desired_ratio
            new_y_min = center_y - desired_height / 2
            new_y_max = center_y + desired_height / 2
        elif current_ratio < desired_ratio:
            desired_width = (new_y_max - new_y_min) * desired_ratio
            new_x_min = center_x - desired_width / 2
            new_x_max = center_x + desired_width / 2

        new_pixel_size = (new_x_max - new_x_min) / self.width
        if new_pixel_size >= self.pixel_threshold:
            self.zoom_level += 1
            self.max_iter = int(self.base_iter * (2 ** self.zoom_level))
        else:
            print("[Main] Detail surpasses resolution. Not increasing iteration.")

        self.x_min = new_x_min
        self.x_max = new_x_max
        self.y_min = new_y_min
        self.y_max = new_y_max

        self.start_calculation()


if __name__=="__main__":
    app=QApplication(sys.argv)
    window=FractalWindow()
    window.show()
    sys.exit(app.exec_())
