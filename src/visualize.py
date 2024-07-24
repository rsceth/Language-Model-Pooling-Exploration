import os
import numpy as np

import matplotlib.pyplot as plt
from cuml.decomposition import PCA
from cuml.manifold import TSNE

def PCA_visualize(outputDir, epoch_i, weight):
    """
        weight visualize during training by epochs
        ==================================================================
        args: outputDir <str>, 
                epoch_i <int>, 
                weight <list>
        return: None
    """
    figdir = os.path.join(outputDir, "epochs_"+str(epoch_i)+".png") 
    # change list operation 
    mean_embedding_vectors = []
    for weight_vec in weight:
        for i in weight_vec:
            mean_embedding_vectors.append(i)

    mean_embedding_vectors = np.array(mean_embedding_vectors)

    pca = PCA(n_components= 2) #reduce down to 50 dim
    y = pca.fit_transform(mean_embedding_vectors)

    # y = TSNE(n_components=2).fit_transform(y) # further reduce to 2 dim using t-SNE

    x, y = y.T
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x, y, s=0.1, c=x, cmap=plt.cm.plasma)
    ax.set_title('PCA plot for end baseline_embeddings weight ', fontsize=18)
    fig.savefig(figdir)