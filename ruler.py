#
# Ruler Estimation
# Author: Abhir Bhalerao, Department of Computer Science, University of Warwick, Coventry, UK
# 28th September, 2021
# abhir.bhalerao@warwick.ac.uk
#
# This class is an approximate ruler graduations/spacing estimator based on the published work:
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
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops

import cv2

class Ruler:

    def __init__(self, block): # constructor


        # must be square
        assert(block.shape[0]==block.shape[1])

        self.block = block.astype('float')
        self.block_size = block.shape[0]

        self.angle = 0
        self.spacing = 0
        self.offset = 0

        return

    def auto_correlate(self, block):

        # this is slow
        # corr = signal.correlate2d(block, block, boundary='fill', mode='samse')

        # use dft
        block_dft = np.fft.fft2(block)
        corr = np.fft.fftshift(np.fft.ifft2(np.multiply(block_dft, block_dft.conj())).real)

        return corr

    def gaussian_window(self, block_size):

        x = np.linspace(-block_size//2, block_size//2-1, block_size)
        y = np.linspace(-block_size//2, block_size//2-1, block_size)
        xv, yv = np.meshgrid(x, y)

        sigma = block_size/5.0

        return np.exp( - 0.5 * (np.square(xv) + np.square(yv))/np.square(sigma))

    # estimate angle first
    def orientation(self):

        self.corr = self.auto_correlate(self.block)
        self.window = self.gaussian_window(self.block_size)

        dft = np.fft.fft2(np.multiply(self.window, self.corr))
        dft_centred = np.abs( np.fft.fftshift(dft) )

        # use region ops orientation estimator
        # need to provide a label map (threholded amp spectrum)

        self.label = np.zeros(dft_centred.shape, dtype='uint8')
        thresh = np.max(dft_centred) * 0.10 # 10% threshold on amplitude spectrum
        self.label[dft_centred>thresh] = 1 # set label to 1

        props = regionprops(self.label, dft_centred)

        # props[0] is the first and only component
        self.angle = np.deg2rad(np.rad2deg(props[0]['orientation']))+np.pi/2

        while (self.angle<0.0):
            self.angle += 2 * np.pi

        return self.angle


    # must be called after angle estimate
    def wavelength(self, which=0):

        def SD(x, t, W):
            assert(t>=0)
            dt = 0.0
            for j in range(W):

                jj = j + t
                if (jj<x.size):
                    dt += np.square(x[j] - x[jj])
            return dt

        def CSD(x, t):
            if (t==0):
                return 1.0
            else:
                W = x.size
                norm = 0.0
                for t2 in range(1,t+1):
                    norm += SD(x, t2, W)
                norm /= t
                return SD(x, t, W)/norm


        def CSDF(x, tmax):
            csdf = np.zeros(tmax+1, dtype='float')
            for t in range(tmax+1):
                csdf[t] = CSD(x, t)
            return csdf


        self.angle_degrees = np.rad2deg(self.angle)

        # use cv2 to do getRotation of auto corr back by ruler angle
        rot_mat = cv2.getRotationMatrix2D((self.block_size//2, self.block_size//2), -self.angle_degrees, 1.0)
        self.rotated = cv2.warpAffine(self.corr, rot_mat, (self.corr.shape[1], self.corr.shape[1]))

        self.spatial = np.sum(self.rotated, axis=0) # column sums
        self.csdf = CSDF(self.spatial, tmax=self.spatial.shape[0]//2)

        # find minima of CSDF
        self.csdf_inv = np.max(self.csdf)-self.csdf
        max_peak_val = np.max(self.csdf_inv)

        self.peaks, _ = find_peaks(self.csdf_inv, height=0.75 * max_peak_val)

        self.spacing = self.peaks[which] # take the first one as default (which=0)

        return self.spacing # wavelenght in pixels

    def sine_model(self, block_size, freq, angle, phase=0):

        y = np.linspace(-block_size//2, block_size//2+1, block_size)
        x = np.linspace(-block_size//2, block_size//2+1, block_size)

        xv, yv = np.meshgrid(x, y)

        r =  xv * np.cos(angle) - yv * np.sin(angle)

        sine_wave = np.sin( freq * np.pi * 2 * r / block_size + phase )

        return sine_wave


    # do this last after angle and wavelength
    def phase_method1(self, invert= -1):

        x = np.linspace(-np.pi, np.pi, self.block_size)
        y = np.linspace(-np.pi, np.pi, self.block_size)
        xv, yv = np.meshgrid(x, y)

        freq = self.block_size/self.spacing

        sine = np.sin( freq * yv )
        cosine = np.cos( freq * yv )

        self.grad_angle_degrees = self.angle_degrees + 90
        rot_mat = cv2.getRotationMatrix2D((self.block_size//2, self.block_size//2), self.grad_angle_degrees, 1.0)
        self.sine_rotated = cv2.warpAffine(sine, rot_mat, (sine.shape[0], sine.shape[1]))
        self.cosine_rotated = cv2.warpAffine(cosine, rot_mat, (cosine.shape[0], cosine.shape[1]))

        # window the basis functions
        self.sine_rotated = np.multiply(self.window, self.sine_rotated)
        self.cosine_rotated = np.multiply(self.window, self.cosine_rotated)

        # if invert = -1, image black on white
        #windowed = np.multiply(self.window, invert * self.block)

        # do quarature by inner products
        sine_ipdt = np.dot(self.sine_rotated.ravel(), invert * self.block.ravel())
        cosine_ipdt = np.dot(self.cosine_rotated.ravel(), invert * self.block.ravel())

        phase = math.atan2(sine_ipdt, cosine_ipdt)

        self.offset = self.spacing * phase/(2 * np.pi)

        return self.offset

    # do this last after angle and wavelength
    def phase(self, invert= -1):


        freq = self.block_size/self.spacing

        self.sine_rotated = self.sine_model(self.block_size, freq, self.angle, 0)
        self.cosine_rotated = self.sine_model(self.block_size, freq, self.angle, np.pi/2)

        # if invert image white on black
        inverted = invert * self.block

        #print(sine_rotated.shape, windowed.shape)
        # do quarature by inner products
        sine_ipdt = np.dot(self.sine_rotated.ravel(), inverted.ravel())
        cosine_ipdt = np.dot(self.cosine_rotated.ravel(), inverted.ravel())

        phase = math.atan2(sine_ipdt, cosine_ipdt)

        self.offset = self.spacing * phase/(2 * np.pi)

        return self.offset

    # helper to do plotting of ruler onto a block
    def show_grads(self, ax, block_size, angle, spacing, offset):

        num_grads = np.int(block_size/spacing)+3
        points = np.zeros((2, 2), dtype='float')

        width = block_size//3
        j = 0

        ruler_angle = angle
        grad_angle = angle + np.pi/2

        for i in range(-num_grads//2,num_grads//2):

            # walk along the ruler directions from bottom to top
            x0 = block_size//2 + (i * spacing + offset) * np.cos(ruler_angle)
            y0 = block_size//2 - (i * spacing + offset) * np.sin(ruler_angle)

            if (x0>=0 and x0<block_size and y0>=0 and y0<block_size):
                ax.plot(x0, y0, 'o', color='orange')

            points[0][0] = x0 + width * np.cos(grad_angle)
            points[0][1] = y0 - width * np.sin(grad_angle)
            points[1][0] = x0 - width * np.cos(grad_angle)
            points[1][1] = y0 + width * np.sin(grad_angle)

            if (points[0][0]>=0 and points[0][0]<block_size) and (points[0][1]>=0 and points[0][1]<block_size):
                if (points[1][0]>=0 and points[1][0]<block_size) and (points[1][1]>=0 and points[1][1]<block_size):
                    ax.plot(points[:,0], points[:,1], 'r', lw=2)
