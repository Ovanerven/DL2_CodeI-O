# first line: 1
@memory.cache
def cached_kmeans_fit(data, n_clusters):
    """Cached version of KMeans fitting for large datasets"""
    print(f"Computing KMeans with n_clusters={n_clusters}")
    start_time = time()

    # For large datasets, use init='k-means++', max_iter=300, n_init=10
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42,
        # Use all available cores
    )

    # If dataset is very large, consider using mini-batch KMeans instead
    # from sklearn.cluster import MiniBatchKMeans
    # kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', batch_size=1024, random_state=42)

    labels = kmeans.fit_predict(data)

    print(f"Computation took {time() - start_time:.2f} seconds")
    return kmeans, labels
