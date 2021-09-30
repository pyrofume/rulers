# Ruler Estimation
# Author: Abhir Bhalerao, Department of Computer Science, University of Warwick, Coventry, UK
# 28th September, 2021
# abhir.bhalerao@warwick.ac.uk
#
# This has some helper functions for preprocessing and extracting ruler blocks from an image
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
import numpy as np
from scipy.ndimage import gaussian_filter


def hp_filter(im):

    factor = im.shape[0]/1024.0 # rule of thumb
    im_smooth = gaussian_filter(im.astype('float'), sigma=factor * 2.5)
    im_hp = im.astype('float') - im_smooth
    im_hp2 = gaussian_filter(im_hp, sigma= factor * 0.5)

    return im_hp2


def extract_block(im, x, y, block_size):


    # pad by blockize//2 on all sides
    pad_i = block_size//2+1
    pad_j = block_size//2+1

    im_padded = np.pad(im, ((pad_i, pad_i), (pad_j, pad_j)), mode='constant', constant_values=(0,0))

    # extract without fear of falling off end
    block = im_padded[pad_i+y-block_size//2:pad_i+y+block_size//2,pad_j+x-block_size//2:pad_j+x+block_size//2]

    return block
