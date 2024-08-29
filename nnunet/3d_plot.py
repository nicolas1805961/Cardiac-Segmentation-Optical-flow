from mayavi import mlab
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load your 3D segmentation volume data (e.g., NIfTI format)
data_path = r'custom_quorum_RV_removed\patient093\patient093_ed_gt.nii.gz'
data = nib.load(data_path)

# Convert the loaded data to a NumPy array
segmentation_array = data.get_fdata()
segmentation_array = segmentation_array[:, :, 2:]

#segmentation_array[segmentation_array == 3] = 4
#segmentation_array[segmentation_array == 2] = 3
#segmentation_array[segmentation_array == 1] = 10

# Get spacing from the header (pixel spacing in x, y, z dimensions)
x_spacing, y_spacing, z_spacing = data.header.get_zooms()
spacing = (x_spacing, y_spacing, z_spacing)

# Get unique labels (excluding background 0)
labels = np.unique(segmentation_array)
labels = labels[labels != 0]  # Exclude background if needed

print(labels)

# Define colors for each label (could be improved by using a color map)
colors = [(1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Extend as needed

transparency = [1.0, 1.0, 1.0, 1.0]

# Iterate over each unique label and create an isosurface
for i, label in enumerate(labels):
    color = colors[i]  # Cycle through colors if more labels than colors are defined
    plt.imshow((segmentation_array == label).astype(int)[:, :, -1])
    plt.show()
    #src = mlab.pipeline.scalar_field((segmentation_array == label).astype(int))
    #src.spacing = spacing  # Set spacing to avoid distortion in visualization
    ## Extract only the voxels matching the current label
    #iso = mlab.pipeline.iso_surface(src, contours=[1.0], color=color, opacity=transparency[i])
    ##iso.actor.property.opacity = 0.3  # Adjust opacity for better visibility

# Show the plot
#mlab.show()
