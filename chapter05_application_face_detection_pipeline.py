# -*- coding: utf-8 -*-
# chapter05_application_face_detection_pipeline.py
"""
In this section, we will take a look at one such feature extraction technique, the Histo‐
gram of Oriented Gradients (HOG), which transforms image pixels into a vector rep‐
resentation that is sensitive to broadly informative image features regardless of
confounding factors like illumination. We will use these features to develop a simple
face detection pipeline, using machine learning algorithms and concepts we’ve seen
throughout this chapter.
"""
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

#%%

"""
The Histogram of Gradients is a straightforward feature extraction procedure that
was developed in the context of identifying pedestrians within images. HOG involves
the following steps:
1. Optionally prenormalize images. This leads to features that resist dependence on
variations in illumination.
2. Convolve the image with two filters that are sensitive to horizontal and vertical
brightness gradients. These capture edge, contour, and texture information.
3. Subdivide the image into cells of a predetermined size, and compute a histogram
of the gradient orientations within each cell.
4. Train a linear SVM classifier on these samples.
5. For an “unknown” image, pass a sliding window across the image, using the
model to evaluate whether that window contains a face or not.
6. If detections overlap, combine them into a single window.
"""

#%%
# example
from skimage import data, color, feature
import skimage.data

image = color.rgb2gray(data.chelsea())
hog_vec, hog_vis = feature.hog(image, visualize=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('input image')

ax[1].imshow(hog_vis)
ax[1].set_title('visualization of HOG features');

#%%
# obtain negative training samples
from skimage import data, transform

imgs_to_use = [
    'camera', 'text', 'coins', 'moon',
    'page', 'clock', 'immunohistochemistry',
    'chelsea', 'coffee', 'hubble_deep_field'
]
images = [
    color.rgb2gray(getattr(data, name)())
    for name in imgs_to_use
]

#%%
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people()
positive_patches = faces.images
positive_patches.shape


#%%
from sklearn.feature_extraction.image import PatchExtractor

def extract_patches(img, N, scale = 1.0,
                    patch_size = positive_patches[0].shape):
    extracted_patch_size = \
        tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size = extracted_patch_size,
                               max_patches = N, random_state = 0)
    patches = extractor.transform(img[np.newaxis])
    
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size)
                            for patch in patches])
    return patches

#%%
negative_patches = np.vstack([extract_patches(im, 1000, scale)
                              for im in images for scale in [0.5, 1.0, 2.0]])
print(negative_patches.shape)

#%%
# visualize them
fig, ax = plt.subplots(6, 10)
for i, axi in enumerate(ax.flat):
    axi.imshow(negative_patches[500 * i], cmap = 'gray')
    axi.axis('off')
    
    
#%%
# Combine sets and extract HOG features
from itertools import chain
X_train = np.array([feature.hog(im)
                    for im in chain(positive_patches,
                                    negative_patches)])
y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1

#%%
print(X_train.shape)

#%%
# training a support vecctor machine
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

print(cross_val_score(GaussianNB(), X_train, y_train))

#%%
'''
We see that on our training data, even a simple naive Bayes algorithm gets us upwards of 90% accuracy. Let's try the support vector machine, with a grid search over a few choices of the C parameter:
'''
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
grid.fit(X_train, y_train)
print(grid.best_score_)

#%%
print(grid.best_params_)

#%%
# lets take the best estimtor and re-train it on the full dataset
model = grid.best_estimator_
model.fit(X_train, y_train)


#%%
# find faces in a enw images
'''
Now that we have this model in place, let's grab a new image and see
how the model does. We will use one portion of the astronaut image for
simplicity (see discussion of this in Caveats and Improvements), and
run a sliding window over it and evaluate each patch:
'''
test_image = skimage.data.astronaut()
test_image = skimage.color.rgb2gray(test_image)
test_image = skimage.transform.rescale(test_image, 0.5)
test_image = test_image[:160, 40:180]

plt.imshow(test_image, cmap = 'gray')
plt.axis('off')

#%%
'''
Next, let's create a window that iterates over patches of
this image, and compute HOG features for each patch:
'''
def sliding_window(img, patch_size = positive_patches[0].shape,
                   istep = 2, jstep = 2, scale = 1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch


indices, patches = zip(*sliding_window(test_image))
patches_hog = np.array([feature.hog(patch) for patch in patches])
print(patches_hog.shape)

#%%
# Finally, we can take these HOG-featured patches and
# use our model to evaluate whether each patch contains a face:
labels = model.predict(patches_hog)
print(labels.sum())


#%%
fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')

Ni, Nj = positive_patches[0].shape
indices = np.array(indices)

for i, j in indices[labels == 1]:
    ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                               alpha=0.3, lw=2, facecolor='none'))

