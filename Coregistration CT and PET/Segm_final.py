import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import matplotlib.pyplot as plt

class IndexTracker:
    def __init__(self, ax, X):
        self.ax = ax
        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='jet')
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_title(f'slice {self.ind}')
        self.im.axes.figure.canvas.draw()

ct = sitk.ReadImage('CT.nii')
pet = sitk.ReadImage('PET_Corrected.nii')

#%% Show loaded

ct_array = sitk.GetArrayFromImage(ct)
pet_array = sitk.GetArrayFromImage(pet)

# Determine a random slice index
slice_index = 80  # Assuming the slices are along the first dimension

# Plot the random slice
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(ct_array[slice_index, :, :], cmap='jet')
plt.title('CT - Slice {}'.format(slice_index))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(pet_array[slice_index, :, :], cmap='jet')
plt.title('PET - Slice {}'.format(slice_index))
plt.axis('off')

plt.tight_layout()
plt.show()

#%% Seed
seed = (310, 288, 100)

# Create an empty segmentation image
seg = sitk.Image(pet.GetSize(), sitk.sitkUInt8)
seg.CopyInformation(pet)

# Initialize the seed
seg[seed] = 1    
seg = sitk.BinaryDilate(seg, [5]*3)

plt.figure()
plt.imshow(ndimage.rotate(sitk.GetArrayFromImage(seg).transpose(), angle=-90)[:, :, 100], cmap='jet')
plt.show()


#%% Overlay the initial seed with the original PET image
seed_overlay = sitk.LabelOverlay(pet, seg, opacity=0.5)#), backgroundValue=255)
seed_overlay_array = sitk.GetArrayFromImage(seed_overlay)
print("Shape of seed_overlay_array:", seed_overlay_array.shape)

img2 = seed_overlay_array[:, :, :, 0]
img2 = img2.transpose()
img2 = ndimage.rotate(img2, angle=-90)

# Visualization
fig3, ax3 = plt.subplots(1, 1)
tracker = IndexTracker(ax3, img2)
fig3.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

#%% Povezivanje regiona
seg_CTRG = sitk.ConnectedThreshold(pet, seedList=[seed], lower=220, upper=5000)
CTRG_overlay = sitk.LabelOverlay(pet, seg_CTRG, opacity=0.5, backgroundValue=1)
CTRG_overlay_array = sitk.GetArrayFromImage(CTRG_overlay)
print("Shape of CTRG_overlay_array:", CTRG_overlay_array.shape)

img3 = CTRG_overlay_array[:, :, :, 0]
img3 = img3.transpose()
img3 = ndimage.rotate(img3, angle=-90)

# Visualization
fig4, ax4 = plt.subplots(1, 1)
tracker = IndexTracker(ax4, img3)
fig4.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

#%% Prikaz slice kao u 3d sliceru
temp3 = ndimage.rotate(pet_array.transpose(), angle=-90)

slice_index = 105
plt.figure()
#plt.imshow(ndimage.rotate(ct_array.transpose(), angle=-90)[:, :, slice_index], cmap='jet')
plt.imshow(temp3[:, :, slice_index]/(np.max(temp3[:, :, slice_index])-np.min(temp3[:, :, slice_index])),alpha = 0.5, cmap = 'Reds_r')
plt.imshow(img3[:, :, slice_index]/(np.max(img3[:, :, slice_index]) - np.min(img3[:, :, slice_index])), alpha = 0.1, cmap='binary')
plt.title('Moving Image 3D - Slice {}'.format(slice_index))
plt.show()

#%% Double scroll - to burn your PC
class IndexTracker2:
    def __init__(self, ax, base_image, overlay_image, alpha=0.2):
        self.ax = ax
        self.base_image = base_image
        self.overlay_image = overlay_image
        self.alpha = alpha
        rows, cols, self.slices = base_image.shape
        self.ind = self.slices // 2

        self.im_base = ax.imshow(self.base_image[:, :, self.ind], cmap='binary')
        self.im_overlay = ax.imshow(self.overlay_image[:, :, self.ind], cmap='Reds_r', alpha=self.alpha)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im_base.set_data(self.base_image[:, :, self.ind])
        self.im_overlay.set_data(self.overlay_image[:, :, self.ind])
        self.ax.set_title(f'Slice {self.ind}')
        self.im_base.axes.figure.canvas.draw()


fig, ax = plt.subplots(figsize=(8, 8))
tracker = IndexTracker2(ax, img3, temp3)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

#%% Get SUV
import json

with open('metadata.json', 'r') as f:
    metadata = json.load(f)

dose = metadata['dosage']
weight = metadata['body_weight']
injection_time = metadata['injection_time']
scan_time = metadata['scan_time']
calibration_factor = metadata['calibration_factor'] / 1000

# Vremenska korekcija
injection_time_sec = int(injection_time[:2]) * 3600 + int(injection_time[2:4]) * 60 + float(injection_time[4:])
scan_time_sec = int(scan_time[:2]) * 3600 + int(scan_time[2:4]) * 60 + float(scan_time[4:])
time_diff_min = (scan_time_sec - injection_time_sec) / 60
fdg_half_life_min = 109.77

decay_correction_factor = np.power(2, -time_diff_min/fdg_half_life_min)

# Aktivnost Bq / mL
roi_mask = CTRG_overlay_array[:,:,:,0] < 1000
roi_pet_values = pet_array[roi_mask]

volume = np.prod(pet.GetSpacing()) / 1000
activity = roi_pet_values / volume / calibration_factor

SUVbw = activity * weight / (decay_correction_factor * dose)

# Calculate min, max, and mean SUVbw in the ROI
min_SUVbw = np.min(SUVbw)
max_SUVbw = np.max(SUVbw)
mean_SUVbw = np.mean(SUVbw)

print(f'Min SUVbw in ROI: {min_SUVbw}')
print(f'Max SUVbw in ROI: {max_SUVbw}')
print(f'Mean SUVbw in ROI: {mean_SUVbw}')
#
