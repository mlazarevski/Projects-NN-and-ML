import os
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import json

# Ucitavanje podataka
def read_dicom_files_in_folder(folder):
    
    files = os.listdir(folder)
    pet_slices = []
    ct_slices = []
    data = None

    for file in files:
        file_path = os.path.join(folder, file)
        try:
            dicom = pydicom.dcmread(file_path, stop_before_pixels=True)
            modality = dicom.Modality
            image = sitk.ReadImage(file_path, sitk.sitkFloat32)
            if modality == 'CT':
                ct_slices.append(image)                
            elif modality == 'PT':
                pet_slices.append(image)
                if not data:
                    data = pydicom.dcmread(file_path)
        except:
            continue

    return data, ct_slices, pet_slices

# Odabir pacijenta
dicom_folder = "DICOM/24032010/09250000"
#dicom_folder = "DICOM/22112909/02360000"
#dicom_folder = "DICOM/22112908/59060000"

data, fixed_image, moving_image = read_dicom_files_in_folder(dicom_folder)
#fixed_image = fixed_image[1:]

#Filter the physical space that we need
z_max = moving_image[0].GetOrigin()[2]
z_min = moving_image[-1].GetOrigin()[2]
fixed_image = [img for img in fixed_image if z_min <= img.GetOrigin()[2] <= z_max]

fixed_image_array = [sitk.GetArrayFromImage(img).squeeze() for img in fixed_image]
fixed_image_array = np.stack(fixed_image_array, axis=0)

moving_image_array = [sitk.GetArrayFromImage(img).squeeze() for img in moving_image]
moving_image_array = np.stack(moving_image_array, axis=0)

#%% Skrol prikaz
class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].set_title('Fixed Image')
ax[1].set_title('Moving Image')

tracker_fixed = IndexTracker(ax[0], fixed_image_array)
tracker_moving = IndexTracker(ax[1], moving_image_array)

fig.canvas.mpl_connect('scroll_event', tracker_fixed.onscroll)
fig.canvas.mpl_connect('scroll_event', tracker_moving.onscroll)
plt.show()



#%% Meta podaci
dosage = getattr(data, 'RadiopharmaceuticalInformationSequence', [{}])[0].get('RadionuclideTotalDose', None) 
body_weight = getattr(data, 'PatientWeight', None) * 1000
injection_time = getattr(data, 'RadiopharmaceuticalInformationSequence', [{}])[0].get('RadiopharmaceuticalStartTime', None)
scan_time = getattr(data, 'AcquisitionTime', None)
calibration_factor = getattr(data, 'DoseCalibrationFactor', None)

print(dosage, body_weight, injection_time, scan_time, calibration_factor)

metadata = {
    'dosage': dosage,
    'body_weight': body_weight,
    'injection_time': injection_time,
    'scan_time': scan_time,
    'calibration_factor': calibration_factor
}

with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

#%% Iz array u 3d format
fixed_image_3d = sitk.GetImageFromArray(fixed_image_array)
fixed_image_3d.SetOrigin(fixed_image[0].GetOrigin())  # Set the origin
fixed_image_3d.SetSpacing(fixed_image[0].GetSpacing()) # Set the spacing

moving_image_3d = sitk.GetImageFromArray(moving_image_array)
moving_image_3d.SetOrigin(moving_image[0].GetOrigin())  # Set the origin
moving_image_3d.SetSpacing(moving_image[0].GetSpacing()) # Set the spacing

print(fixed_image[0].GetSpacing(), moving_image[0].GetSpacing())

#%% Spacing and Origin
fixed_origin = fixed_image[0].GetOrigin()
fixed_spacing = fixed_image[0].GetSpacing()
moving_origin = moving_image[0].GetOrigin()
moving_spacing = moving_image[0].GetSpacing()
print(fixed_origin, moving_origin)
print(fixed_spacing, moving_spacing)

# Extract SliceThickness from the DICOM dataset (assuming all slices have the same thickness)
slice_thickness = getattr(data, 'SliceThickness', None)
#slice_thickness = abs
# Update the spacing to include the slice thickness
if slice_thickness is not None:
    fixed_spacing =  (fixed_spacing[0],  fixed_spacing[1],  slice_thickness)
    moving_spacing = (moving_spacing[0], moving_spacing[1], slice_thickness)

# Set origin and spacing for the fixed image
fixed_image_3d.SetOrigin(fixed_origin)
fixed_image_3d.SetSpacing(fixed_spacing)

# Set origin and spacing for the moving image
moving_image_3d.SetOrigin(moving_origin)
moving_image_3d.SetSpacing(moving_spacing)
#%% Transofrmacija
metric_values = []

def registration_callback(registration_method):
    metric_value = registration_method.GetMetricValue()
    metric_values.append(metric_value)

initial_transform = sitk.CenteredTransformInitializer(fixed_image_3d, moving_image_3d, sitk.AffineTransform(fixed_image_3d.GetDimension()))

moving_resampled = sitk.Resample(moving_image_3d, fixed_image_3d, initial_transform, sitk.sitkLinear)


# Prikaz rezultata inicijalno transformisane moving_resampled slike - prikazati moving_resampled (konvertovan u array) 11. slajs u crnobeloj paleti i odgovarajuci 
#slajs fiksne slike sa transparencijom 0.5, u "jet" paleti 

temp1 = sitk.GetArrayFromImage(moving_resampled)
#%%
print(moving_resampled.GetSpacing())
print(fixed_image_3d.GetSpacing())
print(moving_image_3d.GetSpacing())

plt.figure(figsize=(10, 10))  # You can adjust the figure size as needed

plt.subplot(2, 1, 1)
plt.imshow(temp1[:, :, 200], cmap='gray')

plt.subplot(2, 1, 2)
plt.imshow(fixed_image_array[:, :, 200], cmap='gray')

plt.show()

#%% Inicijalizacija metode registracije: promenljivoj registration_method dodeliti vrednost funkcije sitk.ImageRegistrationMethod()

registration_method = sitk.ImageRegistrationMethod()
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10)


registration_method.SetOptimizerScalesFromPhysicalShift()
registration_method.AddCommand(sitk.sitkIterationEvent, lambda: registration_callback(registration_method))
registration_method.SetInitialTransform(initial_transform)
registration_method.SetInterpolator(sitk.sitkLinear)
final_tranform = registration_method.Execute(fixed_image_3d, moving_image_3d)

# Crtanje metrike sličnosti metric_values u zavisnosti od iteracija pomocu plt.plot funkcije

plt.figure(3)
plt.plot(metric_values)
plt.show()

moving_transformed = sitk.Resample(moving_image_3d, fixed_image_3d, final_tranform, sitk.sitkLinear)
temp2 = sitk.GetArrayFromImage(moving_transformed)

#%%
slice_index = 110

plt.figure(4)
plt.imshow(temp2[slice_index,:,:], cmap = 'gray')
plt.imshow(fixed_image_array[slice_index,:,:], alpha = 0.2, cmap = 'jet')
plt.show()

#%%

slice_index = 110

plt.figure(figsize=(10, 10))  # You can adjust the figure size as needed

plt.subplot(2, 1, 1)
plt.imshow(temp2[slice_index,:,:], cmap = 'gray')

plt.subplot(2, 1, 2)
plt.imshow(fixed_image_array[slice_index,:,:], cmap = 'gray')

plt.show()

#%% Cuvanje za segmentaciju

sitk.WriteImage(fixed_image_3d, 'CT.nii')
sitk.WriteImage(moving_transformed, 'PET_Corrected.nii')
