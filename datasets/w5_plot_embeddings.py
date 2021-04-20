# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

# Read the detections of all the cameras
with open('tracks_seq_c010.pkl', 'rb') as f:
    tracks_seq = pkl.load(f)

descriptors = []
targets = []
for frame in tracks_seq:
    for t in frame: 
        descriptors.append(t.feature_vec[0]/np.max(t.feature_vec[0]))
        targets.append(int(t.id))
classes = np.unique(targets)
print(classes)
tsne = manifold.TSNE(n_components=2, n_iter=3000).fit_transform(descriptors)

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)

# for every class, we'll add a scatter plot separately
for label in classes:
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(targets) if l == label]

    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)

    np.random.seed(int(label))
    c = list(np.random.choice(range(int(256)), size=3))
    color = np.array([int(c[2]), int(c[1]), int(c[0])]).reshape(1,3)
    
    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=color/255.0, label=label)

# build a legend using the labels we set previously
ax.legend(loc='best')

# finally, show the plot
plt.show()