#
# Ruler Estimation
# Author: Abhir Bhalerao, Department of Computer Science, University of Warwick, Coventry, UK
# 28th September, 2021
# abhir.bhalerao@warwick.ac.uk
#
# This is the interactive driver code for the ruler.py class
#
#    Usage:
#        $ python ruler-interactive.py <image-filename>
#
#    Once running, you can click on a point in the display window and that should extract an image block and
#    show the results in the plots window. You can use the fast-keys, [], to increase/decrease the block size
#    (these are trapped by the figure_on_press method).
#    To exit, you need to close the figure window first, then any key will exit the program.  OpenCV windows and matplotlib windows don't play well
#    together, unfortunately.
#
# The algorithms implemented in ruler.py are from our paper:
#
# Ruler Detection for Autoscaling Forensic Images, Abhir Bhalerao and Greg Reynolds.
# International Journal of Digital Crime and Forensics,
# Volume 6, Issue 1, 2014. Pages 9-27.
# https://www.researchgate.net/publication/264277138_Ruler_Detection_for_Autoscaling_Forensic_Images
#
# For the wavelength/spacing estimation, it uses the YIN pitch detector -- please see de Cheveign´e, A. and Kawahara, H. (2002).
# YIN, a fundamental frequency estimator for speech and music. Journal of the Acoustical Society of America, 111(4):1917–1930.
#
# The code is meant to be illustrative and not definitive.
#
# THIS CODE IS PROVIDED “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) SUSTAINED BY YOU OR A THIRD PARTY,
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT ARISING IN ANY WAY OUT OF THE USE OF THIS SAMPLE CODE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
#
# If you do use this code, please leave these notices and/or acknowledge the original code/cite our paper,
# many thanks and we hope you find it useful!
import cv2
import sys

from preproc import *
from ruler import *

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

    global img, block, block_size
    global ax1, ax2, ax3, ax4, ax5, ax6

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print('point is ', x, y)


        # displaying the coordinates
        # on the image window
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, str(x) + ',' +
        #            str(y), (x,y), font,
        #            1, (255, 0, 0), 2)
        # cv2.rectangle(img, (x-block_size//2, y-block_size//2), (x+block_size//2, y+block_size//2), (0, 255, 0), 2)

        copy = img.copy()

        start_point = (x-block_size//2, y-block_size//2)
        end_point = (x+block_size//2, y+block_size//2)
        print(start_point, end_point)

        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(copy, start_point, end_point, color, thickness)
        cv2.imshow('image', copy)

        block = extract_block(hp, x, y, block_size)
        ax1.imshow(block)
        ruler = Ruler(block)

        angle = ruler.orientation()
        spacing = ruler.wavelength()
        offset = ruler.phase(invert= -1)

        print('ruler angle is ', np.rad2deg(angle))
        print('spacing is ', spacing, ' pixels ')

        #ax2.imshow(ruler.corr)
        ax2.imshow(ruler.rotated)

        ax3.clear()
        ax3.plot(ruler.spatial)
        ax3.set_title('spatial')


        ax4.clear()
        ax4.imshow(ruler.block, cmap='gray')
        ruler.show_grads(ax4, ruler.block_size, ruler.angle, ruler.spacing, ruler.offset)
        ax4.set_title('w = {:3.1f}'.format(spacing))

        ax5.imshow(ruler.sine_rotated)

        ax6.clear()
        ax6.plot(ruler.csdf_inv)
        ax6.plot(ruler.peaks, ruler.csdf_inv[ruler.peaks], "x")
        ax6.set_title('YIN')

        plt.draw()

def figure_on_press(event):

    global block_size, max_block_size

    print('figure key pressed was ', event.key)
    sys.stdout.flush()

    if event.key == ']':
        block_size += 16
        if (block_size>max_block_size):
            block_size = max_block_size

    if event.key == '[':
        block_size -= 16
        if (block_size<8):
            block_size = 8

    # checking for right mouse clicks
    # if event==cv2.EVENT_RBUTTONDOWN:
        # blah

# driver function
if __name__=="__main__":

    #print('args:', sys.argv)

    if (len(sys.argv)==0):
        print('Usage: ', sys.argv[0], ' <image filename>')

    else:
        filename = sys.argv[1]

        # reading the image
        img = cv2.imread(filename, 1)

        if (img is None):
            print('image ', filename, ' not found!')
        else:

            # displaying the image
            cv2.imshow('image', img)

            # setting mouse hadler for the image
            # and calling the click_event() function
            cv2.setMouseCallback('image', click_event)

            # preproc
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hp = hp_filter(gray)
            block_size = 128
            max_block_size = 512

            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(10,6),
                        gridspec_kw={'width_ratios': [2, 1, 3], 'height_ratios': [1, 2]})
            fig.canvas.set_window_title('Rulers')
            fig.canvas.mpl_connect('key_press_event', figure_on_press)

            ax1.imshow(np.zeros((block_size,block_size)))
            ax1.set_title('block')
            ax2.imshow(np.zeros((block_size,block_size)))
            ax2.set_title('rotated')
            ax3.plot(np.zeros((1,block_size)))

            ax4.imshow(np.zeros((block_size,block_size)))
            ax5.imshow(np.zeros((block_size,block_size)))
            ax5.set_title('sine')
            ax6.plot(np.zeros((1,block_size)))

            fig.tight_layout()

            plt.show()

            # wait for a key to be pressed to exit
            cv2.waitKey(0)

            # close all subplots
            plt.close('all')

            # close the window
            cv2.destroyAllWindows()
